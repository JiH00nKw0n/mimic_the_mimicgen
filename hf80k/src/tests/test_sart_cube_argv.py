#!/usr/bin/env python3
"""큐브 쌓기의 생성 명령이 SART를 넣기 전과 한 글자도 다르지 않은지 확인한다.

왜 필요한가. SART 증강을 넣으면서 생성 단계의 편수 계산과 건너뛰기 판정을 손봤다. 그
손질이 큐브 쪽 실행을 바꾸지 않았다는 것은 말로 주장할 것이 아니라 기계가 확인할 일이다.
여기 적힌 명령줄과 환경변수는 SART를 넣기 전의 코드에서 그대로 받아 적은 것이다.

같이 확인하는 것이 하나 더 있다. 생성 프로세스의 환경변수에 LAB_SART_로 시작하는 이름이
하나도 없어야 한다. 증강 설정이 생성 쪽으로 새면 그것만으로 생성 동작이 달라진다.

    python3 src/tests/test_sart_cube_argv.py
"""
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, ".."))
PKG = os.path.normpath(os.path.join(SRC, ".."))
ASSETS = os.path.join(PKG, "assets")

# SART를 넣기 전 코드가 만들던 명령줄. {WORK}와 {CHUNK}만 실행할 때 채워진다.
GOLDEN_CMD = [
    "/workspace/isaaclab/isaaclab.sh", "-p", "{WORK}/generate_lab.py",
    "--task", "Isaac-Stack-Cube-LabFR3-HF80K-Fwd-IK-Rel-Mimic-v0",
    "--headless", "--device", "cpu",
    "--num_envs", "16",
    "--generation_num_trials", "500",
    "--input_file", "{WORK}/source_filtered.hdf5",
    "--output_file", "{CHUNK}/gen.part00.hdf5",
]

# SART를 넣기 전 코드가 부모 환경 위에 얹던 값들. {ASSETS}, {SRC}, {WORK}, {CHUNK}만
# 실행할 때 채워진다.
GOLDEN_ENV = {
    "ACCEPT_EULA": "Y",
    "PRIVACY_CONSENT": "Y",
    "OMNI_KIT_ACCEPT_EULA": "YES",
    "PYTHONUNBUFFERED": "1",
    "CUDA_VISIBLE_DEVICES": "0",
    "PHYSICS_PROFILE": "robust_stochastic",
    "LAB_TABLE_USD": "/opt/hf80k/assets/table_scene.usdc",
    "LAB_ROBOT_SPAWN_ROT": "0,0,1,0",
    "LEROBOT_SITE": "",
    "LAB_SYSID_BUNDLE_ROOT": "{ASSETS}/fr3_cube_system_calibration_bundle_v1",
    "LAB_PHYS_OBJECTS": "cube_1,cube_2,cube_3",
    "LAB_PHYS_SURFACE": "work_surface",
    "LAB_PHYS_ARM_ACTUATOR": "a1",
    "LAB_PHYS_GRIPPER_ACTUATOR": "gripper",
    "LAB_PHYS_OBJECT_SIZE": "0.0507",
    "LAB_PHYS_OBJECT_MASSES": ('{"cube_1": 0.0633333333, "cube_2": 0.0650333333, '
                               '"cube_3": 0.0660333333}'),
    "PYTHONPATH": "{SRC}/env:{SRC}/render",
    "LAB_ARM_SCALE": "0.5",
    "LAB_KEEP_FAILED": "0",
    "LAB_SUBTASK_OFFSETS": "10,20",
    "LAB_PROVENANCE_INPUT": "{WORK}/source_filtered.hdf5",
    "LAB_PROVENANCE_OUT": "{CHUNK}/gen.part00.provenance.json",
    "LAB_GEN_SEED": "4200000",
}

PAYLOAD = r'''
import json, os, sys
sys.path.insert(0, os.environ["HF80K_TEST_SRC"])
import orchestrate as orch

cfg = orch.load_config()
captured = {}


def fake_run_parallel(jobs, procs, log, log_path):
    captured["jobs"] = jobs
    return 1.0


orch.run_parallel = fake_run_parallel
orch.already_has_demos = lambda path, want: False

chunk = {"chunk_index": 0, "profile": "nominal_lab", "profile_id": 0,
         "episodes": 500, "seed": 42000,
         "dir": os.path.join(cfg["work_dir"], "chunk_00000")}
os.makedirs(chunk["dir"], exist_ok=True)
try:
    orch.stage_generate(cfg, chunk, lambda msg: None,
                        os.path.join(cfg["work_dir"], "stage.log"))
except Exception:
    pass                       # 산출물 확인에서 멈추는 것이 정상이다. 명령줄만 본다.

job = captured["jobs"][0]
base = dict(os.environ)
# 부모 환경과 값이 다른 이름만 모으면, 부모에 이미 같은 값이 있던 이름이 빠진다.
# 그래서 받아 적어 둔 이름은 값을 그대로 꺼내고, 그 밖에 새로 얹힌 이름만 따로 센다.
watched = json.loads(os.environ["HF80K_TEST_KEYS"])
values = {k: job["env"].get(k) for k in watched}
extra = sorted(k for k, v in job["env"].items() if base.get(k) != v and k not in watched)
print(json.dumps({"n_jobs": len(captured["jobs"]), "cmd": job["cmd"],
                  "env": values, "extra": extra}))
'''

failures = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(("  통과  " if ok else "  실패  ") + label + ((" | " + detail) if detail else ""))
    if not ok:
        failures.append(label)


work = tempfile.mkdtemp(prefix="cubeargv_")
chunk_dir = os.path.join(work, "chunk_00000")
env = dict(os.environ)
for name in list(env):
    if name.startswith(("LAB_", "SART_")) or name in ("PYTHONPATH", "SUBTASK_OFFSETS"):
        env.pop(name)
env.update({
    "HF80K_TEST_SRC": SRC, "TASK_PROFILE": "cube_stack_fr3", "WORK_DIR": work,
    "HF_TOKEN": "", "HF_REPO_ID": "", "LEROBOT_SITE": "", "HF80K_LEROBOT_PATH": "",
    "CHUNK_SIZE": "500", "NUM_ENVS": "16", "GEN_PROCS": "1", "SEED_BASE": "42000",
    "CUDA_VISIBLE_DEVICES": "0", "PHYSICS_PROFILE": "robust_stochastic",
    "HF80K_TEST_KEYS": json.dumps(sorted(GOLDEN_ENV)),
})
proc = subprocess.run([sys.executable, "-c", PAYLOAD], env=env, capture_output=True, text=True)
if proc.returncode != 0:
    print("  실패  생성 단계를 흉내 내지 못했다")
    print("    " + proc.stderr.strip().replace("\n", "\n    ")[-2000:])
    sys.exit(1)
doc = json.loads(proc.stdout.strip().splitlines()[-1])


def fill(text: str) -> str:
    return (text.replace("{WORK}", work).replace("{CHUNK}", chunk_dir)
                .replace("{ASSETS}", ASSETS).replace("{SRC}", SRC))


print("명령줄")
want_cmd = [fill(x) for x in GOLDEN_CMD]
got_cmd = doc["cmd"]
check("생성 프로세스가 하나다", doc["n_jobs"] == 1, "%d개" % doc["n_jobs"])
check("명령줄의 항목 수가 같다", len(got_cmd) == len(want_cmd),
      "받아 적은 것 %d개, 지금 %d개" % (len(want_cmd), len(got_cmd)))
for i, want in enumerate(want_cmd):
    got = got_cmd[i] if i < len(got_cmd) else "(없다)"
    if got != want:
        check("%d번째 항목" % i, False, "받아 적은 것 %r, 지금 %r" % (want, got))
if got_cmd == want_cmd:
    check("명령줄이 받아 적은 것과 완전히 같다", True)

print("\n환경변수")
got_env = doc["env"]
for name, want in sorted(GOLDEN_ENV.items()):
    check("%s" % name, got_env.get(name) == fill(want),
          "받아 적은 것 %r, 지금 %r" % (fill(want), got_env.get(name)))
check("받아 적지 않은 환경변수가 새로 붙지 않았다", not doc["extra"], str(doc["extra"]))
check("LAB_SART_로 시작하는 이름이 하나도 없다",
      not [k for k in list(got_env) + doc["extra"] if k.startswith("LAB_SART_")])

print("\n생성 스크립트 사본")
gen_src = "/workspace/isaaclab/scripts/imitation_learning/isaaclab_mimic/generate_dataset.py"
if not os.path.isfile(gen_src):
    print("  건너뜀  컨테이너 밖이라 Isaac Lab 생성 스크립트가 없다")
else:
    payload = r'''
import os, sys, hashlib
sys.path.insert(0, os.environ["HF80K_TEST_SRC"])
import orchestrate as orch
cfg = orch.load_config()
path = orch.prepare_generate_script(cfg, lambda msg: None)
print(hashlib.sha256(open(path, "rb").read()).hexdigest())
'''
    out = subprocess.run([sys.executable, "-c", payload], env=env,
                         capture_output=True, text=True)
    digest = out.stdout.strip().splitlines()[-1] if out.returncode == 0 else ""
    body = open(gen_src).read()
    inject = "".join("\nimport %s" % n for n in
                     ["lab_register", "clean_success_hook", "provenance_hooks"])
    lines, done = [], False
    for line in body.splitlines():
        lines.append(line)
        if not done and line.startswith("import isaaclab_mimic.envs"):
            lines.append(inject.strip("\n"))
            done = True
    import hashlib
    want_digest = hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest()
    check("큐브용 generate_lab.py가 예전과 같은 내용이다", digest == want_digest,
          "%s vs %s" % (digest[:12], want_digest[:12]))

print()
if failures:
    print("어긋난 항목 %d개: %s" % (len(failures), ", ".join(failures)))
    sys.exit(1)
print("큐브 쌓기의 생성 명령이 SART를 넣기 전과 같다")
