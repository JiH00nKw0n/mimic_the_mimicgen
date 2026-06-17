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
from dataclasses import dataclass
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
# Task profiles.
# We run two examples from the tutorial. A "profile" bundles everything that
# differs between them so the step scripts can stay generic and you just pass
# `--profile franka` or `--profile gr1t2`.
#
#   franka : single-arm Franka stacking cubes. The provided dataset is RAW
#            human demos, so we annotate it ourselves (step 2).
#   gr1t2  : bimanual GR-1 humanoid pick-and-place (left arm picks, right arm
#            places). NVIDIA provides an ALREADY-annotated dataset, so step 2
#            is skipped. This env also needs the Pinocchio kinematics library
#            (`enable_pinocchio`), which the Isaac Lab scripts load specially.
# ----------------------------------------------------------------------------
_S3 = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/IsaacLab/Mimic"


@dataclass(frozen=True)
class TaskProfile:
    name: str
    base_task: str            # playable env (record / replay / video)
    mimic_task: str           # env carrying Mimic subtask metadata (annotate / generate)
    source_url: str           # dataset downloaded in step 1
    source_file: str          # local filename for that download
    annotated_file: str       # dataset step 3 consumes (== source_file when pre-annotated)
    generated_file: str       # full-run output
    generated_small_file: str  # sanity-run output
    pre_annotated: bool       # True -> the download is already annotated (skip step 2)
    enable_pinocchio: bool    # True -> load Pinocchio + register the GR1T2 task modules
    num_envs: int             # default number of parallel envs for generation
    cam_eye: str              # default video camera position 'x,y,z' (close-up framing)
    cam_lookat: str           # default video camera target 'x,y,z'


TASKS: dict[str, TaskProfile] = {
    "franka": TaskProfile(
        name="franka",
        base_task="Isaac-Stack-Cube-Franka-IK-Rel-v0",
        mimic_task="Isaac-Stack-Cube-Franka-IK-Rel-Mimic-v0",
        source_url=f"{_S3}/franka_stack_datasets/dataset.hdf5",
        source_file="source_dataset.hdf5",
        annotated_file="annotated_dataset.hdf5",
        generated_file="generated_dataset.hdf5",
        generated_small_file="generated_dataset_small.hdf5",
        pre_annotated=False,
        enable_pinocchio=False,
        num_envs=10,
        cam_eye="1.0,0.55,0.5",
        cam_lookat="0.45,0.0,0.06",
    ),
    "gr1t2": TaskProfile(
        name="gr1t2",
        base_task="Isaac-PickPlace-GR1T2-Abs-v0",
        mimic_task="Isaac-PickPlace-GR1T2-Abs-Mimic-v0",
        source_url=f"{_S3}/pick_place_datasets/dataset_annotated_gr1.hdf5",
        source_file="gr1t2_annotated_dataset.hdf5",
        annotated_file="gr1t2_annotated_dataset.hdf5",  # already annotated
        generated_file="gr1t2_generated_dataset.hdf5",
        generated_small_file="gr1t2_generated_dataset_small.hdf5",
        pre_annotated=True,
        enable_pinocchio=True,
        num_envs=20,
        cam_eye="0.0,-1.1,1.45",
        cam_lookat="0.0,-0.25,0.95",
    ),
}


def get_profile(name: str) -> TaskProfile:
    """Look up a task profile by name, with a friendly error if it's unknown."""
    if name not in TASKS:
        sys.exit(f"[ERROR] unknown profile '{name}'. Choose from: {', '.join(TASKS)}")
    return TASKS[name]


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
