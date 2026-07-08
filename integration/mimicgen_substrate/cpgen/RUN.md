# CP-Gen ⊕ MimicGen — 재현 레시피 (arpa)

CP-Gen 자체 repo를 **별도 venv**에 설치해 robosuite MimicGen 생태계(scale-setting `SquareWide`)에서
크기 일반화 데모를 생성한다. 우리 작동 중인 `robosuite_mimicgen/venv`는 **건드리지 않는다**.

위치: `arpa:~/mimicgen_jihoonkwon/cpgen_stack/`

## 0. 클론 (`clone.sh`)
`cpgen`, `cpgen-envs`, `robosuite@enable-scale-setting-arena`(=`robosuite_scale`) 세 repo.

## 1. 패치 (`patch_cpgen.py`) — curobo/nerfstudio 없이 CPU로 돌리려는 최소 수술
- `demo_aug/configs/base_config.py`: `from nerfstudio.utils.rich_utils import CONSOLE` → `rich.console.Console`.
- `demo_aug/generate.py`: 최상위 curobo import 2개를 `try/except`(mink 경로에선 불필요).
- `demo_aug/envs/motion_planners/__init__.py`: curobo monkey-patch 전용이라 **비운다**(백업 `.bak`).
- `demo_aug/envs/motion_planners/indexed_configuration.py`: `IndexedConfiguration.__init__`이
  `super().__init__()`을 스킵해서 mink가 기대하는 `self._frame_id_cache`가 없다 → **`self._frame_id_cache = {}` 한 줄 추가**.
  (이건 `install_cpgen.sh` 밖에서 별도로 넣었다 — 재현 시 잊지 말 것.)

## 2. 설치 (`install_cpgen.sh`) — 버전 지뢰 3개가 핵심
- venv(py3.10) + **CPU torch/torchvision를 매칭 버전으로**(`--index-url .../whl/cpu`, force-reinstall).
  robomimic가 안 맞는 torchvision을 끌어오면 `torchvision::nms does not exist`로 터진다.
- `robosuite_scale`(1.5.1), `cpgen-envs`, `robomimic@9273f9c`, `cpgen`(-e), `mink`, `rich`.
- **`mimicgen`도 이 venv에 설치**(cpgen-envs가 import함).
- **`pip install mujoco==3.2.6` 강제**. forked robosuite 1.5.1이 옛 `mj_fullM(m, dst, M)` 시그니처로
  호출하는데 mujoco 3.3+는 `(m, d, dst)`로 바뀌어 `TypeError: mj_fullM(): incompatible`로 터진다.
  (pip이 "robosuite는 ≥3.3.0, mink는 ≥3.8.1 요구" 경고를 내지만 무시 — 런타임은 3.2.6로 동작.)
- **`mink==1.2.0`**(`_resolve_frame_id`가 `_frame_id_cache`를 씀; 위 패치와 짝).

## 3. 소스 데이터
cpgen 자체 annotated square. git-lfs 없이 HF resolve URL로 직접:
```
curl -fL "https://huggingface.co/datasets/cpgen/datasets-src/resolve/main/datasets/source/square.hdf5" \
     -o datasets-src/datasets/source/square.hdf5
ln -sfn ~/mimicgen_jihoonkwon/cpgen_stack/datasets-src/datasets cpgen/src   # src/source/square.hdf5 해석되게
```

## 4. 생성 (mink 대신 `eef_interp` = mujoco 3.2.6와 충돌 없는 CPU 경로)
```bash
cd ~/mimicgen_jihoonkwon/cpgen_stack/cpgen && source ../venv/bin/activate && export MUJOCO_GL=egl
echo "gripper0_right_right_gripper:SquareNut_main,SquareNut_main:peg1" | python demo_aug/generate.py \
  --cfg.demo-path src/source/square.hdf5 --cfg.env-name SquareWide \
  --cfg.motion-planner-type eef_interp --cfg.demo-segmentation-type distance-based \
  --cfg.override-interactions --cfg.no-download-demo --cfg.n-demos 20 --cfg.save-dir ../out/run1
```
- **interaction은 stdin으로** 넘긴다(대화형 `input()` 회피). body 이름이 정확해야 함:
  nut = `SquareNut_main`, square peg = `peg1`(round peg는 peg2), gripper = `gripper0_right_right_gripper`.
- `distance-based` 세그먼트 = LLM/OpenAI 불필요.
- 출력: `out/run1/{successes,failures}/*.hdf5` + `cpgen/datasets/generated/SquareWide/videos/*.mp4`.

## 5. 검증 (`analyze_cpgen.py`)
성공/실패 데모의 nut geom size를 model XML에서 파싱 → 크기가 다양한지(=크기 일반화 진짜인지),
성공/실패가 크기와 어떻게 갈리는지 확인. → `squarewide_run_stats.json`.
