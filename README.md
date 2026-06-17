# mimic_the_mimicgen

NVIDIA **Isaac Lab Mimic** (MimicGen) hands-on: take 10 human teleoperation
demos of a Franka arm stacking cubes and automatically multiply them into ~1000
synthetic demos, then view the result.

This repo holds the small "runner" scripts and notes. The heavy lifting (Isaac
Sim + Isaac Lab) runs inside an NVIDIA Docker container on a remote AWS GPU
server. Nothing big is installed on your laptop.

## How the pieces fit together

```
  YOUR LAPTOP                         REMOTE GPU SERVER (ssh arpa-a6000)
  -----------                         ---------------------------------
  robot_data/mimic_the_mimicgen  ──git push──►  ~/mimicgen_jihoonkwon/mimic_the_mimicgen
        ▲                                              │ runs scripts 00..04
        │ rsync (datasets, outputs)                    ▼
        └───────────────────────────────  docker container "isaac-lab-base"
                                           (Isaac Sim 5.1 + Isaac Lab live here)
```

- **Code** moves through git.
- **Data** (HDF5 datasets, MP4 videos, plots) is large, so it is mirrored back
  to your laptop with `rsync` (`setup/sync_from_remote.sh`), not git.
- The remote server is documented in `robot_data/how_to_use_aws.md`
  (how to connect) and `robot_data/how_to_use_isaac_in_aws.md` (how to run this).

## The pipeline

| Step | Script | What it does | Where it runs |
|------|--------|--------------|---------------|
| 0 | `scripts/00_setup_container.py` | Build + start the Isaac Lab container | remote host |
| 1 | `scripts/01_download_dataset.py` | Download the 10 human demos (HDF5) | remote host |
| 2 | `scripts/02_annotate.py` | Auto-annotate subtask boundaries | container |
| 3 | `scripts/03_generate.py --mode small` | Sanity check: generate 10 demos | container |
| 3 | `scripts/03_generate.py --mode full` | Real run: generate ~1000 demos | container |
| 4 | `scripts/04_record_video.py` | Render replayed demos to MP4 (headless) | container |

Then on your **laptop**:

| Step | Command | What it does |
|------|---------|--------------|
| sync | `bash setup/sync_from_remote.sh` | Pull datasets + videos down |
| 4b | `python3 scripts/04b_inspect_dataset.py` | Summary + plots (no simulator) |

## Quick start (remote)

```bash
cd ~/mimicgen_jihoonkwon/mimic_the_mimicgen
python3 scripts/00_setup_container.py
python3 scripts/01_download_dataset.py
python3 scripts/02_annotate.py
python3 scripts/03_generate.py --mode small     # quick check
python3 scripts/03_generate.py --mode full      # ~30 min, ~1000 demos
python3 scripts/04_record_video.py              # MP4s under outputs/videos/
```

## Notes
- `scripts/_common.py` holds shared constants (container name, task names, paths)
  and the `docker exec` / `docker cp` helpers used by the step scripts.
- `scripts/_record_video_inproc.py` runs *inside* the container (launched by
  step 4); you do not call it directly.
- `notes/teleop_pipeline.md` is a cheat-sheet of the underlying Isaac Lab
  commands and task names.
- **Cost:** the GPU server bills per hour and is shared. Stop the instance when
  you are done (see `how_to_use_aws.md`).
