"""fr3_cube_system_calibration_bundle_v1 loaders (contact/dynamics/camera).

- contact: byte-compatible with stage2 v2 — genaudit.physics.stage2 loads the
  module directory directly (contact_parameter_contract.yaml +
  posterior_samples.csv).
- dynamics_controller: D405 payload + per-joint armature/damping/friction
  nominal and ranges (the quantities missing from the nucleus fr3.usd scene —
  applied to the lab actuator cfg by warmstart_replay --bundle).
- camera: nominal + measured ranges; per-episode sampling is delegated to the
  bundle's own tools/sample_camera_randomization.py.

Pure stdlib+yaml; safe to import anywhere.
"""
from __future__ import annotations

from pathlib import Path


def load_dynamics(bundle_dir: str | Path) -> dict:
    import yaml

    path = Path(bundle_dir) / "modules/dynamics_controller/nominal_and_ranges.yaml"
    doc = yaml.safe_load(path.read_text())
    params = doc["parameters"]
    out = {
        "joint_order": doc["joint_order"],
        "payload_mass_kg": float(doc["payload"]["mass_kg"]),
        "payload_com_parent_m": [float(v) for v in doc["payload"]["com_parent_m"]],
    }
    for key in ("armature_kg_m2", "static_friction", "dynamic_friction",
                "dynamic_to_static_friction_ratio", "viscous_friction",
                "motor_delay_simulation_steps"):
        if key in params:
            entry = params[key]

            def as_list(value):
                return ([float(v) for v in value]
                        if isinstance(value, (list, tuple)) else [float(value)])

            ranges = entry.get("range", [])
            if ranges and not isinstance(ranges[0], (list, tuple)):
                ranges = [ranges]
            out[key] = {
                "nominal": as_list(entry["nominal"]),
                "range": [[float(a), float(b)] for a, b in ranges],
                "status": entry.get("status", ""),
            }
    return out


def contact_dir(bundle_dir: str | Path) -> str:
    """The contact module doubles as a stage2-v2 contract directory."""
    return str(Path(bundle_dir) / "modules/contact")


def camera_module(bundle_dir: str | Path) -> dict:
    import yaml

    module = Path(bundle_dir) / "modules/camera"
    doc = yaml.safe_load(
        (module / "camera_nominal_measured_ranges.yaml").read_text())
    return {"dir": str(module), "doc": doc,
            "sampler": str(module / "tools/sample_camera_randomization.py")}


if __name__ == "__main__":
    import json
    import sys

    dynamics = load_dynamics(sys.argv[1])
    print(json.dumps({k: v for k, v in dynamics.items()
                      if k != "joint_order"}, indent=1)[:900])
