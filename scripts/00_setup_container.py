#!/usr/bin/env python3
"""
STEP 0 - Bring up the Isaac Lab Docker container on the remote GPU server.

What this does and why
----------------------
Isaac Lab ships an official Docker workflow (`IsaacLab/docker/container.py`).
It builds an image called `isaac-lab-base` on top of NVIDIA's pre-built
`nvcr.io/nvidia/isaac-sim:5.1.0` image, then starts that container in the
background ("detached"). Running everything inside this container means we do
NOT have to match Python / CUDA / Isaac Sim versions by hand - they are all
baked into the image.

This script is meant to run ONCE on the remote server. The slow parts (pulling
the 22.9 GB Isaac Sim base image and building on top of it) may already be done
by the time you read this; in that case `container.py start` is quick because
Docker reuses cached layers.

Prerequisites (already true on our server, but checked here):
  - Docker with the NVIDIA Container Toolkit (so containers can see the GPU).
  - The Isaac Lab repo cloned at ~/mimicgen_jihoonkwon/IsaacLab.

Run it like:
    python3 scripts/00_setup_container.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from _common import CONTAINER_NAME, container_running, ensure_container_dirs, in_container, run

# Where the Isaac Lab repo is cloned on the remote host. `container.py` and the
# docker-compose files live under its `docker/` folder.
ISAACLAB_REPO = Path.home() / "mimicgen_jihoonkwon" / "IsaacLab"


def check_gpu_docker() -> None:
    """Confirm Docker can actually see the GPU before we build anything.

    `--gpus all` exposes the host GPU to the container; if the NVIDIA Container
    Toolkit is misconfigured this command fails and nothing else would work.
    """
    print("[1/4] Checking that Docker can see the GPU ...")
    run(
        ["docker", "run", "--rm", "--gpus", "all",
         "nvcr.io/nvidia/cuda:12.4.1-base-ubuntu22.04", "nvidia-smi", "-L"]
    )


def start_container() -> None:
    """Build + start the Isaac Lab container (idempotent).

    `container.py start`:
      1. builds the `isaac-lab-base` image (reusing cached layers), and
      2. starts the container detached.

    On the very first run it prints an interactive prompt asking whether to
    enable X11 forwarding. We are headless, so we answer "n" by piping it in.
    After the first run the answer is remembered in docker/.container.cfg.
    """
    if container_running():
        print(f"[2/4] Container '{CONTAINER_NAME}' already running - skipping start.")
        return
    print("[2/4] Building + starting the Isaac Lab container ...")
    # We pipe "n\n" to auto-answer the headless X11 prompt. We run from the
    # IsaacLab repo so container.py finds its docker/ files.
    run(
        f"cd {ISAACLAB_REPO} && printf 'n\\n' | python3 docker/container.py start",
    )


def verify() -> None:
    """Sanity check: import Isaac Lab + Isaac Sim inside the container."""
    print("[3/4] Verifying the container can import Isaac Lab / Isaac Sim ...")
    # `./isaaclab.sh -p` runs python with Isaac Lab's environment active.
    in_container("./isaaclab.sh -p -c \"import isaaclab, isaacsim; print('isaac import OK')\"")
    print("[4/4] Creating scratch datasets/ and outputs/ folders in the container ...")
    ensure_container_dirs()


def main() -> int:
    if not ISAACLAB_REPO.exists():
        sys.exit(
            f"[ERROR] Isaac Lab repo not found at {ISAACLAB_REPO}.\n"
            f"        Clone it first:  git clone https://github.com/isaac-sim/IsaacLab.git {ISAACLAB_REPO}\n"
        )
    check_gpu_docker()
    start_container()
    verify()
    print("\nDone. The container is up. Next: python3 scripts/01_download_dataset.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
