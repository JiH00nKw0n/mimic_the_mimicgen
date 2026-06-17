"""
Shared constants and helper functions for the MimicGen hands-on pipeline.

Big picture
-----------
We are reproducing NVIDIA's "Isaac Lab Mimic" tutorial. The goal is to take a
small set of human teleoperation demonstrations (10 demos of a Franka arm
stacking cubes) and automatically multiply them into ~1000 synthetic demos.

Where things run
----------------
Everything heavy runs on a remote AWS GPU server inside a Docker container that
NVIDIA ships ("isaac-lab-base"). Isaac Sim (the simulator) and Isaac Lab (the
robotics framework on top of it) live INSIDE that container, so we never install
them on the host directly. Our small "runner" scripts (00..04) run on the
*host* and drive the container with `docker exec` / `docker cp`.

    host (Ubuntu)  ──docker exec──►  container "isaac-lab-base"
                   ──docker cp───►   /workspace/isaaclab  (Isaac Lab lives here)

Why docker cp?
--------------
The Isaac Lab container only bind-mounts a few of its own folders (source,
scripts, docs, tools) from the host. Our datasets are not in those folders, so
the simplest, most reliable way to move files in/out of the container is the
plain `docker cp` command. This file wraps that so the step scripts stay short.
"""

from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path

# ----------------------------------------------------------------------------
# Paths on the HOST (the remote server where you run scripts 00..04).
# REPO_ROOT is the folder that contains this `scripts/` directory, i.e. the
# checkout of `mimic_the_mimicgen`. We resolve it relative to this file so the
# scripts work no matter what directory you launch them from.
# ----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = REPO_ROOT / "datasets"   # HDF5 files live here on the host
OUTPUTS_DIR = REPO_ROOT / "outputs"     # videos / plots / logs live here

# ----------------------------------------------------------------------------
# The Isaac Lab Docker container.
# `isaac-lab-base` is the container name set by Isaac Lab's docker-compose
# (service "base"). `/workspace/isaaclab` is where Isaac Lab is installed
# inside the container; `./isaaclab.sh` is its launcher script.
# ----------------------------------------------------------------------------
CONTAINER_NAME = "isaac-lab-base"
ISAACLAB_PATH = "/workspace/isaaclab"
# A scratch folder *inside* the container where we stage datasets while the
# Isaac Lab scripts read/write them. (Container-only; we copy results back out.)
CONTAINER_DATA = f"{ISAACLAB_PATH}/datasets"
CONTAINER_OUT = f"{ISAACLAB_PATH}/outputs"

# ----------------------------------------------------------------------------
# Task (environment) names used by the tutorial. Isaac Lab registers many
# "gym" environments; these are the Franka cube-stacking ones.
#   - TASK_BASE: the plain environment you record / replay / play in.
#   - TASK_MIMIC: the same task wrapped with extra Mimic metadata (subtask
#     boundaries) that the MimicGen data generator needs.
# ----------------------------------------------------------------------------
TASK_BASE = "Isaac-Stack-Cube-Franka-IK-Rel-v0"
TASK_MIMIC = "Isaac-Stack-Cube-Franka-IK-Rel-Mimic-v0"


# ----------------------------------------------------------------------------
# Small process helpers.
# ----------------------------------------------------------------------------
def run(cmd: list[str] | str, **kwargs) -> subprocess.CompletedProcess:
    """Run a command on the HOST, echoing it first so you can see what happens.

    Raises if the command fails (check=True) unless you pass check=False.
    """
    printable = cmd if isinstance(cmd, str) else " ".join(shlex.quote(c) for c in cmd)
    print(f"\n$ {printable}\n", flush=True)
    return subprocess.run(cmd, shell=isinstance(cmd, str), check=kwargs.pop("check", True), **kwargs)


def container_running() -> bool:
    """Return True if the `isaac-lab-base` container is up."""
    out = subprocess.run(
        ["docker", "ps", "--filter", f"name=^{CONTAINER_NAME}$", "--format", "{{.Names}}"],
        capture_output=True, text=True,
    )
    return CONTAINER_NAME in out.stdout.split()


def require_container() -> None:
    """Stop with a clear message if the container is not running yet."""
    if not container_running():
        sys.exit(
            f"[ERROR] Container '{CONTAINER_NAME}' is not running.\n"
            f"        Start it first:  python3 scripts/00_setup_container.py\n"
        )


def in_container(bash_command: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a bash command INSIDE the Isaac Lab container.

    We always `cd` into the Isaac Lab folder first, and use a login shell
    (`bash -lc`) so the container's Python environment is set up correctly.
    """
    full = f"cd {ISAACLAB_PATH} && {bash_command}"
    return run(["docker", "exec", CONTAINER_NAME, "bash", "-lc", full], check=check)


def cp_to_container(host_path: str | Path, container_path: str) -> None:
    """Copy a file from the host into the container (docker cp)."""
    run(["docker", "cp", str(host_path), f"{CONTAINER_NAME}:{container_path}"])


def cp_from_container(container_path: str, host_path: str | Path) -> None:
    """Copy a file from the container back to the host (docker cp)."""
    Path(host_path).parent.mkdir(parents=True, exist_ok=True)
    run(["docker", "cp", f"{CONTAINER_NAME}:{container_path}", str(host_path)])


def ensure_container_dirs() -> None:
    """Create the scratch dataset/output folders inside the container."""
    in_container(f"mkdir -p {CONTAINER_DATA} {CONTAINER_OUT}")
