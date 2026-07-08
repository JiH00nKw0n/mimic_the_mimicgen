import sys, runpy
import mimicgen  # noqa: registers Square_D0/D1/D2 robosuite envs
sys.argv = [
    "playback_dataset.py",
    "--dataset", "datasets/generated/square_sart.hdf5",
    "--video_path", "datasets/generated/square_sart_playback.mp4",
    "--n", "6", "--render_image_names", "agentview",
]
runpy.run_path("robomimic/robomimic/scripts/playback_dataset.py", run_name="__main__")
