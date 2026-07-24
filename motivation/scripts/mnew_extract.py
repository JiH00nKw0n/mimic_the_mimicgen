"""Extract AttemptRecords from the merged motivation_new pools (8 tasks x N0/N1/N2)
into jsonl, for the E1 DGR-vs-transform + source-skew analysis. Uses the N2
geometry (bounds_new) as the frozen distance axis, source states from the
annotated source datasets. Runs after the chunk merge completes.
"""
import sys
from pathlib import Path

REPO = "/home/ubuntu/mimicgen_jihoonkwon/mimic_the_mimicgen/motivation"
GEN = Path("/home/ubuntu/mimicgen_jihoonkwon/experiments/motivation_new/gen")
SRC = "/home/ubuntu/mimicgen_jihoonkwon/robosuite_mimicgen/mimicgen/datasets/source"
CFG = f"{REPO}/configs/tasks"
OUT = Path("/home/ubuntu/mimicgen_jihoonkwon/experiments/motivation_new/records")
TASKS = ["square", "threading", "coffee", "three_piece_assembly", "stack",
         "stack_three", "mug_cleanup", "hammer_cleanup"]

sys.path.insert(0, REPO)
from genaudit.config import load_task_spec  # noqa: E402
from genaudit.envs.bounds_new import NEW_BOUNDS  # noqa: E402
from genaudit.factors.initial_condition import build_task_geometry  # noqa: E402
from genaudit.records.extract import (  # noqa: E402
    extract_attempt_records, load_source_initial_states,
)
from genaudit.records.schema import write_jsonl  # noqa: E402


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for task in TASKS:
        spec = load_task_spec(f"{CFG}/{task}.yaml")
        geom = build_task_geometry(task, NEW_BOUNDS[task]["N2"], spec.symmetry_orders)
        objs = list(geom.symmetry_orders)
        src_xy, src_yaw = load_source_initial_states(f"{SRC}/{task}.hdf5", objs)
        for v in ("N0", "N1", "N2"):
            pool = GEN / f"{task}_{v}"
            demo = pool / "demo.hdf5"
            failed = pool / "demo_failed.hdf5"
            try:
                recs = extract_attempt_records(
                    task, v, geom, src_xy, src_yaw,
                    demo_hdf5=demo if demo.exists() else None,
                    failed_hdf5=failed if failed.exists() else None)
                n = write_jsonl(recs, OUT / f"{task}_{v}_attempts.jsonl")
                dgr = sum(r.success for r in recs) / len(recs) if recs else 0
                print(f"{task}_{v}: {n} records, DGR={dgr*100:.0f}%", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"{task}_{v}: ERR {type(e).__name__}: {e}", flush=True)
    (OUT / "EXTRACT_DONE").touch()
    print("EXTRACT DONE", flush=True)


if __name__ == "__main__":
    main()
