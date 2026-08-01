"""Stage2 contact-calibration x MimicGen DGR ablation driver (aidas-l40s).

Arms (per task, IC variant frozen at N2 so physics is the only factor):
  P0_base       robosuite default physics (same-code-path baseline)
  P1_nominal    stage2 nominal mapping (deterministic)
  P2_posterior  per-attempt posterior rows (calibrated-uncertainty DR)
  P3_robust     robust_stochastic 80/20 wide DR + joint rules
  P5_omni       P3 + OmniReset Table-2 actuation DR
  P4_hisrc      P3 physics, hi-DGR sources only (stack {4,7}, square {0,4})

Fixed attempts (guarantee=False, keep_failed uncapped) so DGR denominators are
exact: stack 500/arm, square 900/arm; P4 200/400. 500-attempt chunks, jobs<=5
(coexists with running trainers/renderer), resumable (chunks with
important_stats.json are skipped), disk-guarded.

Stages: prepare | smoke | gen | merge | extract | all  (smoke is standalone)

Run (server):
  cd ~/mimicgen_jihoonkwon/mimic_the_mimicgen/motivation
  V=~/mimicgen_jihoonkwon/robosuite_mimicgen/venv/bin/python
  PYTHONPATH=$PWD $V scripts/stage2_ablation.py --stage smoke
  setsid nohup env PYTHONPATH=$PWD $V scripts/stage2_ablation.py --stage all \
      --jobs 5 </dev/null >> ~/stage2_ablation.log 2>&1 &
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HOME = Path("/home/ubuntu")
M = HOME / "mimicgen_jihoonkwon/mimic_the_mimicgen/motivation"
RS = HOME / "mimicgen_jihoonkwon/robosuite_mimicgen"
TPL = RS / "mimicgen/mimicgen/exps/templates/robosuite"
SRC = RS / "mimicgen/datasets/source"
OUT = HOME / "mimicgen_jihoonkwon/experiments/stage2_ablation"
CONTRACT = HOME / "stage2_contact_calibration_v2"
CHUNK = 500
MIN_FREE_GB = 15

# task -> (attempts/arm, hi-DGR source ids from motivation_new N2 casestudy)
TASKS = {"stack": (500, (4, 7)), "square": (900, (0, 4))}
# arm -> (profile, class suffix, omni, hi_source_only)
ARMS: dict[str, tuple[str | None, str, bool, bool]] = {
    "P0_base": (None, "", False, False),
    "P1_nominal": ("nominal", "s2n", False, False),
    "P2_posterior": ("posterior", "s2p", False, False),
    "P3_robust": ("robust", "s2r", False, False),
    "P5_omni": ("robust", "s2o", True, False),
    "P4_hisrc": ("robust", "s2r", False, True),
}
P4_ATTEMPTS = {"stack": 200, "square": 400}


def phys_seed(task: str, arm: str, chunk_seed: int) -> int:
    return zlib.crc32(f"stage2/{task}/{arm}/{chunk_seed}".encode()) & 0x7FFFFFFF


def chunk_plan(total: int) -> list[tuple[int, int]]:
    out, seed, remaining = [], 1, total
    while remaining > 0:
        n = min(CHUNK, remaining)
        out.append((seed, n))
        remaining -= n
        seed += 1
    return out


def free_gb(path: Path) -> float:
    stat = os.statvfs(path)
    return stat.f_bavail * stat.f_frsize / 1e9


def source_for(task: str, arm: str) -> tuple[Path, str | None]:
    """(source hdf5, filter_key) for an arm."""
    if ARMS[arm][3]:
        return OUT / "sources" / f"{task}_hi.hdf5", "hi"
    return SRC / f"{task}.hdf5", None


def prepare_sources() -> None:
    """P4 filtered source copies: full copy + robomimic mask (originals untouched)."""
    import h5py
    import numpy as np

    (OUT / "sources").mkdir(parents=True, exist_ok=True)
    for task, (_, hi) in TASKS.items():
        dst = OUT / "sources" / f"{task}_hi.hdf5"
        if dst.exists():
            print(f"[prepare] {dst.name} exists, skip")
            continue
        shutil.copy(SRC / f"{task}.hdf5", dst)
        with h5py.File(dst, "a") as handle:
            names = np.array([f"demo_{i}".encode() for i in hi])
            handle.require_group("mask")
            if "hi" in handle["mask"]:
                del handle["mask/hi"]
            handle["mask"].create_dataset("hi", data=names)
        print(f"[prepare] {dst.name}: mask/hi = demos {list(hi)}")


def build_chunk(task: str, arm: str, seed: int, n: int, root: Path) -> tuple[Path, list[str]]:
    """Write mg_config (+physics sidecar); returns (config path, extra argv)."""
    sys.path.insert(0, str(M))
    from genaudit.envs.robosuite_variants import variant_class_name
    from genaudit.generation.mimicgen_backend import (
        build_generation_config, load_template, save_config,
    )

    profile, suffix, omni, _ = ARMS[arm]
    chunk_dir = root / f"{task}_{arm}" / f"chunk_seed{seed}"
    class_name = variant_class_name(task, f"N2{suffix}")
    config = build_generation_config(
        load_template(TPL / f"{task}.json"),
        task_name=class_name,
        source_dataset=str(source_for(task, arm)[0]),
        output_folder=str(chunk_dir),
        num_attempts=n,
        seed=seed,
    )
    filter_key = source_for(task, arm)[1]
    if filter_key:
        config["experiment"]["source"]["filter_key"] = filter_key
    config_path = save_config(config, chunk_dir / "mg_config.json")

    extra: list[str] = ["--auto-remove-exp"]
    if profile is not None:
        sidecar = {
            "task": task, "profile": profile, "suffix": suffix,
            "seed": phys_seed(task, arm, seed),
            "contract_dir": str(CONTRACT), "base_variant": "N2", "omni": omni,
        }
        physics_path = chunk_dir / "physics.json"
        physics_path.write_text(json.dumps(sidecar, indent=2))
        extra += ["--physics", str(physics_path)]
    return config_path, extra


def run_chunk(job: tuple[str, str, int, int], root: Path, log_dir: Path) -> tuple[tuple, str]:
    task, arm, seed, n = job
    chunk_dir = root / f"{task}_{arm}" / f"chunk_seed{seed}"
    if list(chunk_dir.rglob("important_stats.json")):
        return (job, "skip")
    if free_gb(OUT) < MIN_FREE_GB:
        return (job, "DISK_LOW_ABORT")
    try:
        config_path, extra = build_chunk(task, arm, seed, n, root)
        log = log_dir / f"{task}_{arm}_s{seed}.log"
        with open(log, "w") as handle:
            code = subprocess.run(
                [sys.executable, "-m", "genaudit.generation.run_mimicgen",
                 "--config", str(config_path), *extra],
                stdout=handle, stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONPATH": str(M)},
            ).returncode
        if code != 0:
            return (job, f"exit{code}")
        # mimicgen's main() swallows generation exceptions and exits 0;
        # important_stats.json is written only after a fully completed run,
        # and the resume pre-check above guarantees it cannot be stale.
        if not list(chunk_dir.rglob("important_stats.json")):
            return (job, "no_stats(mimicgen swallowed an exception; see log)")
        return (job, "ok")
    except Exception as error:  # noqa: BLE001
        return (job, f"err {type(error).__name__}: {error}")


def stage_gen(jobs_n: int) -> bool:
    root = OUT / "gen"
    (root.parent / "gen/logs").mkdir(parents=True, exist_ok=True)
    jobs = []
    for task, (attempts, _) in TASKS.items():
        for arm in ARMS:
            total = P4_ATTEMPTS[task] if ARMS[arm][3] else attempts
            for seed, n in chunk_plan(total):
                jobs.append((task, arm, seed, n))
    jobs.sort(key=lambda j: (-j[3], j[0], j[1]))
    print(f"[gen] {len(jobs)} chunks across {jobs_n} workers", flush=True)
    with ThreadPoolExecutor(max_workers=jobs_n) as pool:
        results = list(pool.map(lambda j: run_chunk(j, root, root / "logs"), jobs))
    failed = [(j, s) for j, s in results if s not in ("ok", "skip")]
    for job, status in failed:
        print(f"[gen] FAIL {job}: {status}", flush=True)
    (root / "GEN_CHUNKS_DONE.json").write_text(json.dumps(
        {"total": len(results), "failed": len(failed)}, indent=2))
    print(f"[gen] chunks done: {len(results) - len(failed)}/{len(results)} ok", flush=True)
    return not failed


def _merge_files(chunk_files: list[Path], out_path: Path) -> int:
    import h5py

    total = 0
    index = 0
    with h5py.File(out_path, "w") as dst:
        data_group = dst.create_group("data")
        first = True
        for chunk_file in chunk_files:
            with h5py.File(chunk_file, "r") as src:
                if "data" not in src:
                    continue
                # take attrs from the first chunk that carries real env_args
                # (a zero-success chunk's demo.hdf5 can hold empty attrs)
                if first and src["data"].attrs.get("env_args"):
                    for key, value in src["data"].attrs.items():
                        data_group.attrs[key] = value
                    first = False
                names = sorted(src["data"].keys(),
                               key=lambda n: int(n.split("_")[1]))
                for name in names:
                    src.copy(f"data/{name}", data_group, name=f"demo_{index}")
                    total += int(src["data"][name].attrs.get("num_samples", 0))
                    index += 1
        data_group.attrs["total"] = total
    return index


def stage_merge() -> None:
    root = OUT / "gen"
    for task in TASKS:
        for arm in ARMS:
            pool_dir = root / f"{task}_{arm}"
            stats_files = sorted(pool_dir.glob("chunk_seed*/*/important_stats.json"))
            marker = pool_dir / "MERGED"
            if marker.exists():
                # stale-pool guard: if chunks were regenerated after a partial
                # merge, the completed-chunk count changed — remerge.
                try:
                    merged_chunks = int(marker.read_text().strip() or "-1")
                except ValueError:
                    merged_chunks = -1
                if merged_chunks == len(stats_files):
                    print(f"[merge] {pool_dir.name}: cached ({merged_chunks} chunks)")
                    continue
                print(f"[merge] {pool_dir.name}: chunk count changed "
                      f"({merged_chunks} -> {len(stats_files)}), remerging")
            demos = sorted(pool_dir.glob("chunk_seed*/*/demo.hdf5"))
            fails = sorted(pool_dir.glob("chunk_seed*/*/demo_failed.hdf5"))
            if not demos and not fails:
                print(f"[merge] {pool_dir.name}: NO_CHUNKS")
                continue
            n_ok = _merge_files(demos, pool_dir / "demo.hdf5") if demos else 0
            n_fail = _merge_files(fails, pool_dir / "demo_failed.hdf5") if fails else 0
            n_succ = n_att = 0
            for stats_file in stats_files:
                stats = json.loads(stats_file.read_text())
                n_succ += stats["num_success"]
                n_att += stats["num_attempts"]
            (pool_dir / "important_stats.json").write_text(json.dumps({
                "num_success": n_succ, "num_failures": n_att - n_succ,
                "num_attempts": n_att,
                "success_rate": 100 * n_succ / n_att if n_att else 0}))
            marker.write_text(str(len(stats_files)))
            print(f"[merge] {pool_dir.name}: demos={n_ok} fails={n_fail} "
                  f"DGR={100 * n_succ / n_att if n_att else 0:.1f}%", flush=True)


def stage_extract() -> None:
    sys.path.insert(0, str(M))
    from genaudit.config import load_task_spec
    from genaudit.envs.bounds_new import NEW_BOUNDS
    from genaudit.factors.initial_condition import build_task_geometry
    from genaudit.records.extract import (
        extract_attempt_records, load_source_initial_states,
    )
    from genaudit.records.schema import write_jsonl

    records_dir = OUT / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    for task, (_, hi) in TASKS.items():
        spec = load_task_spec(M / "configs/tasks" / f"{task}.yaml")
        geometry = build_task_geometry(task, NEW_BOUNDS[task]["N2"], spec.symmetry_orders)
        objects = list(geometry.symmetry_orders)
        for arm in ARMS:
            pool_dir = OUT / "gen" / f"{task}_{arm}"
            demo = pool_dir / "demo.hdf5"
            failed = pool_dir / "demo_failed.hdf5"
            if not demo.exists() and not failed.exists():
                print(f"[extract] {task}_{arm}: missing pool, skip")
                continue
            # P4 pools index into the FILTERED source list; loading the source
            # states from the filtered copy keeps ids consistent, and
            # orig_source_id restores the 10-source numbering for analysis.
            source_path, filter_key = source_for(task, arm)
            source_xy, source_yaw = load_source_initial_states(source_path, objects)
            if filter_key:
                source_xy = [source_xy[i] for i in hi]
                source_yaw = [source_yaw[i] for i in hi]
            try:
                records = extract_attempt_records(
                    task, f"N2_{arm}", geometry, source_xy, source_yaw,
                    demo_hdf5=demo if demo.exists() else None,
                    failed_hdf5=failed if failed.exists() else None,
                    stage2_extras=True,
                )
            except Exception as error:  # noqa: BLE001
                print(f"[extract] {task}_{arm}: ERR {type(error).__name__}: {error}")
                continue
            if filter_key:
                for record in records:
                    record.extras["orig_source_id"] = hi[record.source_demo_id]
            count = write_jsonl(records, records_dir / f"{task}_{arm}_attempts.jsonl")
            dgr = sum(r.success for r in records) / len(records) if records else 0
            with_extras = sum("s2_cube_mu" in r.extras for r in records)
            print(f"[extract] {task}_{arm}: {count} records DGR={dgr * 100:.1f}% "
                  f"s2-extras={with_extras}", flush=True)
    (records_dir / "EXTRACT_DONE").touch()


def _model_xmls(pool_dir: Path, limit: int = 4) -> list[str]:
    """Up to `limit` model XMLs pooled across BOTH files, so per-attempt
    variation is checkable even when only one attempt succeeded."""
    import h5py

    xmls: list[str] = []
    for name in ("demo.hdf5", "demo_failed.hdf5"):
        path = next(iter(pool_dir.rglob(name)), None)
        if path is None:
            continue
        with h5py.File(path, "r") as handle:
            if "data" not in handle:
                continue
            for demo in list(handle["data"].keys())[: limit - len(xmls)]:
                xml = handle[f"data/{demo}"].attrs["model_file"]
                xmls.append(xml.decode() if isinstance(xml, bytes) else xml)
        if len(xmls) >= limit:
            break
    return xmls


def _numeric(xml: str, key: str) -> float | None:
    for match in re.finditer(r"<numeric\b[^>]*>", xml):
        if f'name="s2_{key}"' in match.group(0):
            data = re.search(r'data="([^"]+)"', match.group(0))
            if data:
                return float(data.group(1).split()[0])
    return None


def _geom_sliding(xml: str, geom_name: str) -> float | None:
    match = re.search(
        rf'<geom\b[^>]*name="{geom_name}"[^>]*friction="([^"\s]+)', xml)
    return float(match.group(1)) if match else None


def stage_smoke(jobs_n: int = 3) -> bool:
    """10-attempt gate for P1/P3/P5 x both tasks; verifies physics reached sim."""
    root = OUT / "smoke"
    log_dir = root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(task, arm, 999, 10) for task in TASKS
            for arm in ("P1_nominal", "P2_posterior", "P3_robust",
                        "P5_omni", "P4_hisrc")]
    with ThreadPoolExecutor(max_workers=jobs_n) as pool:
        results = list(pool.map(lambda j: run_chunk(j, root, log_dir), jobs))
    ok = True
    for (task, arm, seed, _n), status in results:
        if status not in ("ok", "skip"):
            print(f"[smoke] GEN FAIL {task}_{arm}: {status} "
                  f"(see {log_dir}/{task}_{arm}_s{seed}.log)")
            ok = False
    if not ok:
        return False

    for task, arm, seed, _n in jobs:
        pool_dir = root / f"{task}_{arm}" / f"chunk_seed{seed}"
        xmls = _model_xmls(pool_dir)
        if not xmls:
            print(f"[smoke] {task}_{arm}: no demos written FAIL")
            ok = False
            continue
        cube_mu = [_numeric(x, "cube_mu") for x in xmls]
        table_num = _numeric(xmls[0], "table_mu")
        table_attr = _geom_sliding(xmls[0], "table_collision")
        checks = {
            "numerics_present": all(v is not None for v in cube_mu),
            "attr_matches_numeric": (
                table_num is not None and table_attr is not None
                and abs(table_attr - table_num) < 1e-4
            ),
        }
        if arm == "P1_nominal":
            checks["nominal_fixed"] = cube_mu[0] is not None and all(
                v is not None and abs(v - cube_mu[0]) < 1e-9 for v in cube_mu)
        else:
            checks["per_attempt_varies"] = len({round(v, 6) for v in cube_mu if v}) > 1
        if arm == "P5_omni":
            checks["omni_recorded"] = _numeric(xmls[0], "osc_kp_scale") is not None
        status = "PASS" if all(checks.values()) else f"FAIL {checks}"
        print(f"[smoke] {task}_{arm}: {status} "
              f"(table attr={table_attr} num={table_num}, cube_mu={cube_mu})",
              flush=True)
        ok = ok and all(checks.values())
    return ok


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True,
                        choices=["prepare", "smoke", "gen", "merge", "extract", "all"])
    parser.add_argument("--jobs", type=int, default=5)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not CONTRACT.exists():
        sys.exit(f"contract dir missing: {CONTRACT}")

    if args.stage == "prepare":
        prepare_sources()
    elif args.stage == "smoke":
        prepare_sources()
        sys.exit(0 if stage_smoke() else 1)
    elif args.stage == "gen":
        prepare_sources()
        stage_gen(args.jobs)
    elif args.stage == "merge":
        stage_merge()
    elif args.stage == "extract":
        stage_extract()
    elif args.stage == "all":
        prepare_sources()
        if not stage_gen(args.jobs):
            print("[all] gen had failures — merging what completed", flush=True)
        stage_merge()
        stage_extract()
        (OUT / "STAGE2_DONE").touch()
        print("STAGE2 ABLATION DONE", flush=True)


if __name__ == "__main__":
    main()
