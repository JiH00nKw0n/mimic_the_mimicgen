"""Register physics-randomized env variants (stage2 ablation).

Same seam as `robosuite_variants.py`: dynamic subclasses attached to the
mimicgen env modules, so robosuite's name-based registry resolves them and
mimicgen/robosuite sources stay untouched. The parent is a REGISTERED
N-variant class (e.g. Stack_N2), so IC bounds are identical across physics
arms and physics is the only varying factor.

Per-attempt flow (robosuite hard_reset=True, mimicgen calls env.reset() per
attempt): reset() -> _load_model() [we draw a fresh contract-space sample and
mutate the assembled MJCF] -> _initialize_sim() -> _reset_internal() [we apply
the OSC gain scales drawn in the same cycle]. Realized values are written as
<custom><numeric name="s2_*"> elements, so every generated demo's model_file
attr carries its own physics — post-hoc verifiable and extractable.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from genaudit.envs.robosuite_variants import (
    NEW_VARIANT_PARENTS,
    variant_class_name,
)
from genaudit.physics.stage2 import (
    load_contract,
    sample_omni_actuation,
    sample_profile,
    to_mujoco,
)

# geom-name prefixes of the movable task objects (object-object contact class)
OBJECT_GEOM_PREFIXES = {
    "stack": ("cubeA_g", "cubeB_g"),
    "stack_three": ("cubeA_g", "cubeB_g", "cubeC_g"),
    "square": ("SquareNut_g", "RoundNut_g"),
}
# object geoms that are plain centered boxes -> safe for size jitter
SIZE_JITTER_PREFIXES = {
    "stack": ("cubeA_g", "cubeB_g"),
    "stack_three": ("cubeA_g", "cubeB_g", "cubeC_g"),
    "square": (),  # nut hole clearance is tolerance-critical; never scale
}
# static receptacle bodies whose collision geoms take the object mu (square pegs)
RECEPTACLE_BODIES = {"square": ("peg1", "peg2")}
FINGER_GEOM_SUFFIXES = (
    "finger1_collision", "finger2_collision",
    "finger1_pad_collision", "finger2_pad_collision",
)


def _set_sliding(geom, mu: float) -> None:
    parts: list[str] = list((geom.get("friction") or "1 0.005 0.0001").split())
    while len(parts) < 3:
        parts.append("0.0001")
    parts[0] = f"{mu:.6g}"  # torsional/rolling stay at robosuite defaults
    geom.set("friction", " ".join(parts))


def _scale_attr(el, attr: str, scale: float) -> None:
    raw = el.get(attr)
    if raw is None:
        return  # absent attr = 0/default; x-scaling keeps it there
    el.set(attr, " ".join(f"{float(v) * scale:.6g}" for v in raw.split()))


def apply_physics_to_model(root, task: str, mj: dict, omni: dict | None) -> dict:
    """Mutate the assembled MJCF ElementTree. Returns match counts (fail-loud)."""
    counts = {"table": 0, "objects": 0, "fingers": 0,
              "grip_actuators": 0, "receptacles": 0, "robot_joints": 0}

    for geom in root.iter("geom"):
        name = geom.get("name") or ""
        if name == "table_collision":
            _set_sliding(geom, mj["table_sliding"])
            counts["table"] += 1
        elif any(name.startswith(p) for p in OBJECT_GEOM_PREFIXES[task]):
            _set_sliding(geom, mj["object_sliding"])
            density = float(geom.get("density", "1000"))
            geom.set("density", f"{density * mj['mass_scale']:.6g}")
            if any(name.startswith(p) for p in SIZE_JITTER_PREFIXES[task]):
                _scale_attr(geom, "size", mj["size_scale"])
            counts["objects"] += 1
        elif any(name.endswith(s) for s in FINGER_GEOM_SUFFIXES):
            _set_sliding(geom, mj["finger_sliding"])
            counts["fingers"] += 1

    for body in root.iter("body"):
        if body.get("name") in RECEPTACLE_BODIES.get(task, ()):
            for geom in body.findall("geom"):
                if geom.get("group", "0") == "0":  # collision geom
                    _set_sliding(geom, mj["object_sliding"])
                    counts["receptacles"] += 1

    # NOTE: contract cube damping is deliberately NOT applied — MuJoCo free
    # joints take a single 6-dof damping scalar, and the linear-equivalent
    # value over-damps rotation ~1000x (see stage2.py header). Sampled values
    # are still recorded in the s2_* numerics for the integration report.
    for joint in root.iter("joint"):
        name = joint.get("name") or ""
        if omni and name.startswith("robot0_joint"):
            _scale_attr(joint, "damping", omni["joint_damping_scale"])
            # Panda arm joints carry damping only (no armature/frictionloss
            # attrs -> sim nominal 0; x-scaling a 0 nominal keeps 0, so those
            # two Table-2 axes are inert on this robot — documented in report).
            _scale_attr(joint, "armature", omni["joint_armature_scale"])
            _scale_attr(joint, "frictionloss", omni["joint_fric_scale"])
            counts["robot_joints"] += 1
        elif omni and name.endswith(("finger_joint1", "finger_joint2")):
            _scale_attr(joint, "damping", omni["grip_damping_scale"])

    for actuator in root.iter("position"):
        name = actuator.get("name") or ""
        if "gripper_finger_joint" in name:
            _scale_attr(actuator, "forcerange", mj["gripper_force_scale"])
            if omni:
                _scale_attr(actuator, "kp", omni["grip_kp_scale"])
            counts["grip_actuators"] += 1

    required = {"table": 1, "objects": 2, "fingers": 4, "grip_actuators": 2}
    if task in RECEPTACLE_BODIES:
        required["receptacles"] = 1
    missing = {k: v for k, v in required.items() if counts[k] < v}
    if missing:
        names = sorted((g.get("name") or "?") for g in root.iter("geom"))[:80]
        raise RuntimeError(
            f"physics application matched too few elements {missing} "
            f"(counts={counts}); geom names: {names}"
        )
    return counts


def _write_custom_numerics(root, values: dict) -> None:
    """Record realized values as <custom><numeric name='s2_*'> elements."""
    import xml.etree.ElementTree as ET

    custom = root.find("custom")
    if custom is None:
        custom = ET.SubElement(root, "custom")
    for numeric in list(custom.findall("numeric")):
        if (numeric.get("name") or "").startswith("s2_"):
            custom.remove(numeric)
    for key, value in sorted(values.items()):
        ET.SubElement(custom, "numeric",
                      {"name": f"s2_{key}", "data": f"{value:.6g}"})


def register_physics_variant(
    task: str,
    profile: str,
    suffix: str,
    seed: int,
    contract_dir: str | Path,
    base_variant: str = "N2",
    omni: bool = False,
) -> str:
    """Build+register one physics env class; returns its registry name.

    Must run AFTER register_new_variants() (parent is the registered
    {Task}_{base_variant} class). Idempotent per process.
    """
    import importlib

    module_name, _ = NEW_VARIANT_PARENTS[task]
    module = importlib.import_module(module_name)
    parent_name = variant_class_name(task, base_variant)
    if not hasattr(module, parent_name):
        raise RuntimeError(
            f"{parent_name} not registered yet — call register_new_variants() first"
        )
    parent = getattr(module, parent_name)
    class_name = variant_class_name(task, f"{base_variant}{suffix}")
    if hasattr(module, class_name):
        return class_name

    contract = load_contract(contract_dir)
    rng = random.Random(seed)
    task_key = task

    def _load_model(self):
        parent._load_model(self)
        sample = sample_profile(contract, profile, rng)
        omni_sample = sample_omni_actuation(rng) if omni else None
        mj = to_mujoco(sample)
        apply_physics_to_model(self.model.root, task_key, mj, omni_sample)
        record = dict(sample)
        record["force_ratio"] = mj["gripper_force_scale"]
        if omni_sample:
            record.update(omni_sample)
        _write_custom_numerics(self.model.root, record)
        self._s2_sample = sample
        self._s2_omni = omni_sample

    def _reset_internal(self):
        if omni and getattr(self, "_s2_omni", None):
            cfg = self.robots[0].controller_config
            if not hasattr(self, "_s2_base_gains"):
                # robosuite 1.4's OSC reads "damping_ratio"; legacy datasets
                # carry "damping" (same semantics, swallowed by **kwargs) —
                # seed the real key from the legacy value so the scale lands.
                self._s2_base_gains = {
                    "kp": cfg["kp"],
                    "damping_ratio": cfg.get("damping_ratio",
                                             cfg.get("damping", 1.0)),
                }
            for key, base in self._s2_base_gains.items():
                scale = (self._s2_omni["osc_kp_scale"] if key == "kp"
                         else self._s2_omni["osc_damping_scale"])
                cfg[key] = ([v * scale for v in base]
                            if isinstance(base, (list, tuple)) else base * scale)
        parent._reset_internal(self)

    variant_class = type(
        class_name,
        (parent,),
        {
            "_load_model": _load_model,
            "_reset_internal": _reset_internal,
            "__doc__": (
                f"{class_name}: {parent_name} + stage2 contact physics "
                f"(profile={profile}, omni={omni}, seed={seed})."
            ),
            "__module__": module_name,
        },
    )
    setattr(module, class_name, variant_class)
    return class_name


def register_from_file(physics_json: str | Path) -> str:
    """CLI hook for run_mimicgen: register the variant a sidecar describes.

    Sidecar schema: {"task", "profile", "suffix", "seed", "contract_dir",
                     "base_variant"?, "omni"?}
    """
    spec = json.loads(Path(physics_json).read_text())
    return register_physics_variant(
        task=spec["task"],
        profile=spec["profile"],
        suffix=spec["suffix"],
        seed=int(spec["seed"]),
        contract_dir=spec["contract_dir"],
        base_variant=spec.get("base_variant", "N2"),
        omni=bool(spec.get("omni", False)),
    )
