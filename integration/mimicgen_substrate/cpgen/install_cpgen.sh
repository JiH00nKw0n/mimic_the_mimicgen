#!/usr/bin/env bash
set -eo pipefail
cd "$HOME/mimicgen_jihoonkwon/cpgen_stack"
echo "[1/8] patch $(date +%T)"; python3 patch_cpgen.py
echo "[2/8] venv $(date +%T)"; virtualenv -p python3.10 venv; source venv/bin/activate; pip install -q -U pip wheel
echo "[3/8] torch cpu $(date +%T)"; pip install torch --index-url https://download.pytorch.org/whl/cpu 2>&1 | tail -1
echo "[4/8] robosuite_scale $(date +%T)"; pip install -e robosuite_scale 2>&1 | tail -2
echo "[5/8] cpgen-envs $(date +%T)"; pip install -e cpgen-envs 2>&1 | tail -2
echo "[6/8] robomimic pinned $(date +%T)"; pip install "robomimic @ git+https://github.com/ARISE-Initiative/robomimic.git@9273f9c" 2>&1 | tail -2
echo "[7/8] cpgen + mink $(date +%T)"; pip install -e cpgen 2>&1 | tail -2; pip install mink 2>&1 | tail -1
echo "[8/8] import test $(date +%T)"
python - <<PY
import robosuite, cpgen_envs, mink, robomimic
print("robosuite", getattr(robosuite,"__version__","?"))
import demo_aug.configs.base_config as bc
print("cpgen base_config import OK (curobo/nerf patched)")
print("IMPORTS_OK")
PY
echo "INSTALL_DONE $(date +%T)"
