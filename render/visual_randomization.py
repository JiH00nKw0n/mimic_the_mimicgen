"""Visual randomization for the FR3 3-cube RGB export.

Implements the RL team's contract
(`fr3_visual_randomization_handoff_v1_320x180`) inside OUR render loop.

Why a re-implementation rather than importing their `source/events.py`: their
randomizers are IsaacLab manager-based `EventTermCfg` terms that fire on env
reset inside a tiled multi-env RL task. Our renderer is a single-env
state-playback loop with no event manager, so the terms cannot fire. What
transfers is the *contract* — which quantities are randomized, over which
ranges, and at which scope — and that is what this module reproduces, reading
their YAML as the single source of truth so the numbers are never copied by
hand.

Scope contract (`config/visual_randomization_profiles.yaml: scope_contract`):

  profile selection, HDRI + dome light, floor   -> per PROCESS
  camera pose and focal length                  -> per EPISODE
  object and local materials                    -> per EPISODE
  mid-episode changes                           -> forbidden

One deliberate deviation, recorded here and in the export metadata: their
forbidden list bans changing the global dome light per episode *inside a tiled
env*, because one light is shared by every tile. We render one environment at
a time, so that failure mode does not exist for us — but we still follow the
process scope, because matching their episode-to-episode statistics matters
more than the extra diversity we could get.

Pure USD/IsaacLab calls; import only after AppLauncher has started.
"""
from __future__ import annotations

import math
import os

import numpy as np
import yaml

PROFILE_NAMES = ("nominal_lab", "lab_variation", "stress_tail")


def load_config(config_dir: str) -> dict:
    """Read the profile + camera-range YAMLs of the handoff package."""
    with open(os.path.join(config_dir, "visual_randomization_profiles.yaml")) as fh:
        profiles = yaml.safe_load(fh)
    ranges_path = os.path.join(config_dir, "camera_nominal_measured_ranges.yaml")
    with open(ranges_path) as fh:
        cam_ranges = yaml.safe_load(fh)
    return {"profiles": profiles, "camera_ranges": cam_ranges}


def episode_profile_plan(n_episodes: int, mixture: dict, seed: int) -> list[str]:
    """Split n episodes across profiles by the contract mixture (50/40/10).

    Deterministic and exact-count: we allocate by largest remainder rather than
    sampling, so a 25-episode run gets exactly 13/10/2 instead of a binomial
    draw that could land 17/6/2.
    """
    weights = [float(mixture[p]) for p in PROFILE_NAMES]
    raw = [w * n_episodes for w in weights]
    counts = [int(math.floor(v)) for v in raw]
    remainder = n_episodes - sum(counts)
    order = np.argsort([-(raw[i] - counts[i]) for i in range(len(counts))])
    for i in range(remainder):
        counts[order[i % len(counts)]] += 1
    plan = []
    for name, c in zip(PROFILE_NAMES, counts):
        plan.extend([name] * c)
    rng = np.random.default_rng(seed)
    rng.shuffle(plan)
    return plan


def _camera_jitter_spec(cam_ranges: dict) -> dict:
    """role -> {t_ball_m, r_ball_deg, focal_scale(optional)} from measured ranges."""
    out = {}
    local = cam_ranges["measured_ranges"]["camera_local"]
    for role, vals in local["fixed_d435"]["by_role"].items():
        out[role] = {
            "t_ball_m": float(vals["translation_uniform_ball_radius_m"]),
            "r_ball_deg": float(vals["rotation_uniform_vector_ball_radius_deg"]),
        }
    wrist = local["wrist_d405"]
    out["wrist"] = {
        "t_ball_m": float(wrist["translation_uniform_ball_radius_m"]),
        "r_ball_deg": float(wrist["rotation_uniform_vector_ball_radius_deg"]),
        "focal_scale": tuple(float(v) for v in wrist["focal_length_scale_uniform"]),
    }
    return out


def _uniform_ball(rng, radius: float) -> np.ndarray:
    """Uniform sample inside a 3-ball (their `uniform_ball_volume` convention)."""
    v = rng.normal(size=3)
    v /= np.linalg.norm(v)
    return v * radius * rng.random() ** (1.0 / 3.0)


def _rotvec_to_quat_wxyz(rv: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(rv))
    if theta < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = rv / theta
    s = math.sin(theta / 2.0)
    return np.array([math.cos(theta / 2.0), axis[0] * s, axis[1] * s, axis[2] * s])


def _quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


class VisualRandomizer:
    """Applies one profile's visual randomization to a live stage."""

    # our scene's prims, keyed by the package's semantic names
    MATERIAL_TARGETS = {
        "cube_1": ("{ENV}/Cube_1", "gripper_and_cubes"),
        "cube_2": ("{ENV}/Cube_2", "gripper_and_cubes"),
        "cube_3": ("{ENV}/Cube_3", "gripper_and_cubes"),
        "gripper": ("{ENV}/Robot/fr3_hand", "gripper_and_cubes"),
        "table": ("{ENV}/Table", "table"),
        # curtain_side_back / curtain_front have no counterpart in our scene
        # (no room shell); skipped and reported in `self.skipped`.
    }

    def __init__(self, config_dir: str, package_root: str, profile: str, seed: int):
        if profile not in PROFILE_NAMES:
            raise ValueError(f"unknown profile {profile!r}; expected {PROFILE_NAMES}")
        cfg = load_config(config_dir)
        self.profiles_doc = cfg["profiles"]
        self.spec = self.profiles_doc["profiles"][profile]
        self.nominal_colors = self.profiles_doc["nominal_colors_rgb"]
        self.materials = self.profiles_doc["materials"]
        self.floor_spec = self.profiles_doc["floor"]
        self.cam_jitter = _camera_jitter_spec(cfg["camera_ranges"])
        self.package_root = package_root
        self.profile = profile
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.skipped: list[str] = []
        self.applied: dict = {"profile": profile, "seed": self.seed}

    # ---------------------------------------------------------------- process
    def apply_process_scope(self, env_prefix: str = "/World/envs/env_0") -> dict:
        """HDRI + dome light intensity, and the floor material. Once per run."""
        import omni.usd
        from pxr import Gf, UsdGeom, UsdLux

        stage = omni.usd.get_context().get_stage()
        hdri = os.path.join(self.package_root, self.spec["hdri"])
        intensity = float(self.spec["dome_light_intensity"])
        applied = {"hdri": hdri, "dome_light_intensity": intensity}

        dome = None
        for prim in stage.Traverse():
            if prim.IsA(UsdLux.DomeLight):
                dome = prim
                break
        if dome is None:
            dome = UsdLux.DomeLight.Define(stage, "/World/vrandDomeLight").GetPrim()
            applied["dome_created"] = True
        if os.path.isfile(hdri):
            # direct attribute writes: the UsdLux helpers map to different
            # attribute names across USD versions (the handoff's own note)
            attr = dome.GetAttribute("inputs:texture:file")
            if not attr:
                attr = UsdLux.DomeLight(dome).CreateTextureFileAttr()
            attr.Set(hdri)
        else:
            applied["hdri_missing"] = True
            self.skipped.append(f"hdri:{hdri}")
        iattr = dome.GetAttribute("inputs:intensity")
        if not iattr:
            iattr = UsdLux.DomeLight(dome).CreateIntensityAttr()
        iattr.Set(intensity)
        # yaw-only dome rotation: a full random orientation tilts the horizon,
        # which no fixed indoor light rig does
        yaw = float(self.rng.uniform(0.0, 360.0))
        xf = UsdGeom.Xformable(dome)
        xf.ClearXformOpOrder()
        xf.AddRotateZOp().Set(yaw)
        applied["dome_yaw_deg"] = yaw

        floor_color = tuple(float(v) for v in self.floor_spec["fallback_color_rgb"])
        tint = float(self.rng.uniform(*self.floor_spec["diffuse_tint_grayscale"]))
        floor_prim = self._find_floor(stage)
        if floor_prim is not None:
            self._bind_preview_surface(
                floor_prim,
                diffuse=tuple(c * tint for c in floor_color),
                roughness=float(self.rng.uniform(*self.floor_spec["roughness"])),
                metallic=float(self.rng.uniform(*self.floor_spec["metallic"])),
                mat_path="/World/Looks/vrand_floor")
            applied["floor_prim"] = str(floor_prim.GetPath())
        else:
            self.skipped.append("floor")
        self.applied["process"] = applied
        return applied

    def _find_floor(self, stage):
        for path in ("/World/defaultGroundPlane/GroundPlane/CollisionMesh",
                     "/World/defaultGroundPlane", "/World/GroundPlane"):
            prim = stage.GetPrimAtPath(path)
            if prim and prim.IsValid():
                return prim
        return None

    # ---------------------------------------------------------------- episode
    def apply_episode_scope(self, scene, env_prefix: str = "/World/envs/env_0") -> dict:
        """Object/table materials and camera pose+focal jitter. Once per episode."""
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        lo, hi = float(self.spec["color_scales"][0]), float(self.spec["color_scales"][-1])
        colors = {}
        for name, (prim_tmpl, mat_key) in self.MATERIAL_TARGETS.items():
            prim_path = prim_tmpl.replace("{ENV}", env_prefix)
            prim = stage.GetPrimAtPath(prim_path)
            if not prim or not prim.IsValid():
                self.skipped.append(f"material:{name}:{prim_path}")
                continue
            base = np.asarray(self.nominal_colors[name], dtype=float)
            scale = self.rng.uniform(lo, hi, size=3)
            rgb = np.clip(base * scale, 0.0, 1.0)
            m = self.materials[mat_key]
            self._bind_preview_surface(
                prim,
                diffuse=tuple(float(v) for v in rgb),
                roughness=float(self.rng.uniform(*m["roughness"])),
                metallic=float(self.rng.uniform(*m["metallic"])),
                mat_path=f"/World/Looks/vrand_{name}")
            colors[name] = [round(float(v), 4) for v in rgb]

        cams = {}
        for role, spec in self.cam_jitter.items():
            try:
                cam = scene[role]
            except KeyError:
                continue
            dt = _uniform_ball(self.rng, spec["t_ball_m"])
            rv = _uniform_ball(self.rng, math.radians(spec["r_ball_deg"]))
            cams[role] = {"d_translation_m": [round(float(v), 5) for v in dt],
                          "d_rotvec_deg": [round(float(math.degrees(v)), 5) for v in rv]}
            self._jitter_camera(stage, cam, dt, rv)
            if "focal_scale" in spec:
                cams[role]["focal_scale"] = self._scale_focal(
                    stage, cam, spec["focal_scale"])
        out = {"colors": colors, "cameras": cams}
        self.applied["episode"] = out
        return out

    def _jitter_camera(self, stage, cam, d_trans, d_rotvec):
        """Perturb the camera prim's LOCAL transform (parent stays the link)."""
        from pxr import Gf, UsdGeom

        for path in cam.cfg.prim_path.replace("{ENV_REGEX_NS}", "/World/envs/env_0"), :
            prim = stage.GetPrimAtPath(path)
            if not prim or not prim.IsValid():
                self.skipped.append(f"camera:{path}")
                return
            xf = UsdGeom.Xformable(prim)
            ops = xf.GetOrderedXformOps()
            for op in ops:
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    base = np.asarray(op.Get(), dtype=float)
                    op.Set(Gf.Vec3d(*(base + d_trans)))
                elif op.GetOpType() == UsdGeom.XformOp.TypeOrient:
                    q = op.Get()
                    base = np.array([q.GetReal(), *q.GetImaginary()], dtype=float)
                    new = _quat_mul(base, _rotvec_to_quat_wxyz(d_rotvec))
                    op.Set(type(q)(float(new[0]),
                                   type(q.GetImaginary())(*[float(v) for v in new[1:]])))

    def _scale_focal(self, stage, cam, scale_range):
        from pxr import UsdGeom

        s = float(self.rng.uniform(*scale_range))
        path = cam.cfg.prim_path.replace("{ENV_REGEX_NS}", "/World/envs/env_0")
        prim = stage.GetPrimAtPath(path)
        if prim and prim.IsValid():
            camera = UsdGeom.Camera(prim)
            attr = camera.GetFocalLengthAttr()
            if attr:
                attr.Set(float(attr.Get()) * s)
        return round(s, 6)

    def _bind_preview_surface(self, prim, diffuse, roughness, metallic, mat_path):
        """Create and bind a UsdPreviewSurface (version-stable; no Replicator)."""
        import isaaclab.sim as sim_utils

        cfg = sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(diffuse),
                                          roughness=roughness, metallic=metallic)
        cfg.func(mat_path, cfg)
        sim_utils.bind_visual_material(str(prim.GetPath()), mat_path)

    # ------------------------------------------------------------------ report
    def report(self) -> dict:
        out = dict(self.applied)
        out["skipped"] = sorted(set(self.skipped))
        return out
