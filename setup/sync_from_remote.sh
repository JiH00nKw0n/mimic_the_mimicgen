#!/usr/bin/env bash
#
# Pull the generated artifacts (datasets + outputs) from the remote GPU server
# down to this local checkout. RUN THIS ON YOUR LAPTOP.
#
# Division of labor:
#   - CODE  travels through git (you push from one side, pull on the other).
#   - DATA  (HDF5 datasets, MP4 videos, plots) is large and not in git, so we
#           mirror it with rsync here.
#
# We deliberately sync ONLY datasets/ and outputs/ so this never clobbers your
# local code. The heavy environment (IsaacLab clone, Docker, venv) stays on the
# remote and is never copied.
#
# Usage:
#   bash setup/sync_from_remote.sh
#
set -euo pipefail

# SSH host alias from ~/.ssh/config (see how_to_use_aws.md).
REMOTE="arpa-a6000"
REMOTE_REPO="mimicgen_jihoonkwon/mimic_the_mimicgen"

# Resolve this repo's root (the parent of this setup/ folder) regardless of
# where the script is called from.
LOCAL_REPO="$(cd "$(dirname "$0")/.." && pwd)"

echo "Syncing datasets/ and outputs/ from ${REMOTE}:${REMOTE_REPO} ..."
mkdir -p "${LOCAL_REPO}/datasets" "${LOCAL_REPO}/outputs"

# -a archive, -v verbose, -z compress, -P show progress + resume partial files.
rsync -avzP "${REMOTE}:${REMOTE_REPO}/datasets/" "${LOCAL_REPO}/datasets/"
rsync -avzP "${REMOTE}:${REMOTE_REPO}/outputs/"  "${LOCAL_REPO}/outputs/"

echo "Done. Open MP4s in outputs/videos/ or run scripts/04b_inspect_dataset.py."
