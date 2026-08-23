#!/usr/bin/env python3
"""SART를 끄면 파이프라인이 예전과 똑같이 도는지 확인한다.

왜 필요한가. SART 증강은 큐브 쌓기에는 쓰지 않는다. 그런데 단계를 하나 끼워 넣었으므로,
큐브 쪽 실행이 조금이라도 달라지지 않았다는 것을 사람 눈이 아니라 검사가 말해 줘야 한다.

두 가지를 본다. 하나는 큐브 프로필에서 증강 단계가 0.0초를 돌려주고 청크 디렉터리에
파일을 하나도 만들지 않는다는 것이다. 다른 하나는 핀 프로필에서도 SART_ENABLE=0을 주면
같아진다는 것이다. 그리고 두 경우 모두 뒤 단계가 읽을 파일이 원래의 gen.hdf5여야 한다.

    python3 src/tests/test_sart_stage_off.py
"""
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, ".."))

failures = []

PAYLOAD = r'''
import json, os, sys, tempfile
sys.path.insert(0, os.environ["HF80K_TEST_SRC"])
import h5py
import orchestrate as orch

cfg = orch.load_config()
cdir = tempfile.mkdtemp(prefix="sartoff_")
with h5py.File(os.path.join(cdir, "gen.hdf5"), "w") as fh:
    data = fh.create_group("data")
    for i in range(3):
        grp = data.create_group("demo_%d" % i)
        grp.attrs["num_samples"] = 10
        grp.attrs["success"] = True
before = sorted(os.listdir(cdir))
chunk = {"chunk_index": 0, "profile": "nominal_lab", "profile_id": 0,
         "episodes": 500, "seed": 42000, "dir": cdir}
secs = orch.stage_sart(cfg, chunk, lambda msg: None, os.path.join(cdir, "stage.log"))
print(json.dumps({
    "profile": orch.PROFILE.name,
    "sart_enable": bool(cfg["sart_enable"]),
    "seconds": secs,
    "seconds_is_zero": secs == 0.0,
    "files_before": before,
    "files_after": sorted(os.listdir(cdir)),
    "dataset_path": os.path.basename(orch.gen_dataset_path(chunk)),
    "touched_produced": "produced" in chunk,
}))
'''


def run_case(label: str, extra_env: dict) -> None:
    env = dict(os.environ)
    env.update({"HF80K_TEST_SRC": SRC, "WORK_DIR": tempfile.mkdtemp(prefix="sartoff_work_")})
    env.update(extra_env)
    for name in ("HF_TOKEN", "HF_REPO_ID"):
        env.setdefault(name, "")
    proc = subprocess.run([sys.executable, "-c", PAYLOAD], env=env,
                          capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"  실패  {label}: 하위 프로세스가 죽었다")
        print("    " + proc.stderr.strip().replace("\n", "\n    ")[-1500:])
        failures.append(label)
        return
    doc = json.loads(proc.stdout.strip().splitlines()[-1])

    def one(name: str, ok: bool, detail: str = "") -> None:
        print(("  통과  " if ok else "  실패  ") + f"{label}: {name}"
              + ((" | " + detail) if detail else ""))
        if not ok:
            failures.append(f"{label}: {name}")

    one("증강이 꺼져 있다", doc["sart_enable"] is False,
        f"프로필 {doc['profile']}")
    one("정확히 0.0초를 돌려준다", doc["seconds_is_zero"], f"돌려준 값 {doc['seconds']!r}")
    one("청크 디렉터리에 아무 파일도 만들지 않는다",
        doc["files_before"] == doc["files_after"],
        f"{doc['files_before']} -> {doc['files_after']}")
    one("청크의 편수 값을 건드리지 않는다", doc["touched_produced"] is False)
    one("뒤 단계가 읽을 파일이 gen.hdf5다", doc["dataset_path"] == "gen.hdf5",
        doc["dataset_path"])


print("큐브 쌓기 프로필. sart 절이 없으므로 증강이 꺼진다")
run_case("cube_stack_fr3", {"TASK_PROFILE": "cube_stack_fr3", "SART_ENABLE": ""})

print("\n핀 삽입 프로필이지만 SART_ENABLE=0으로 껐다")
run_case("peg_insert_fr3 + SART_ENABLE=0",
         {"TASK_PROFILE": "peg_insert_fr3", "SART_ENABLE": "0"})

print()
if failures:
    print("어긋난 항목 %d개: %s" % (len(failures), ", ".join(failures)))
    sys.exit(1)
print("SART를 끄면 파이프라인이 예전과 똑같이 돈다")
