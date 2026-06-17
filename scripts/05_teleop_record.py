#!/usr/bin/env python3
"""
STEP 5 - Record your own demos by teleoperating over WebRTC.

What this is
------------
Instead of downloading NVIDIA's human demos (step 1), you can make your own:
stream the simulator from the server GPU to your laptop over WebRTC, drive the
robot with your keyboard, and record the result as a demo dataset. Those demos
can then go straight into the same pipeline (annotate -> generate).

How the streaming works
-----------------------
Isaac Lab's launcher supports `--livestream 1` (WebRTC, public network). The
server renders on its GPU and streams the viewport; your laptop runs the free
"Isaac Sim WebRTC Streaming Client" and sends your keyboard/mouse back. Because
the server is on AWS (behind NAT), we tell the stream its public IP via a Kit
setting so the video can reach you.

Before you run this
-------------------
  1. Open the WebRTC ports on the instance's security group (inbound):
       - TCP 49100  (signaling)
       - UDP 47998  (media)
     Source = your laptop's public IP /32.
  2. Download the "Isaac Sim WebRTC Streaming Client" (macOS) from NVIDIA.

Then run this on the server (in tmux), connect the client to the server's public
IP, and teleoperate.

Keyboard controls (printed again when the app starts)
  W / S : move +x / -x (forward / back)      Z / X : roll +/-
  A / D : move +y / -y (left / right)         T / G : pitch +/-
  Q / E : move +z / -z (up / down)            C / V : yaw +/-
  K     : toggle gripper (open/close)
  (a reset key is printed by the recorder on start)

Run it like:
    python3 scripts/05_teleop_record.py --num-demos 5
    python3 scripts/05_teleop_record.py --public-ip 18.191.163.73 --num-demos 5
"""

from __future__ import annotations

import argparse
import subprocess

from _common import (
    CONTAINER_DATA,
    DATASETS_DIR,
    cp_from_container,
    ensure_container_dirs,
    get_profile,
    in_container,
    require_container,
)

# WebRTC ports (Isaac Sim defaults). Only the public IP needs to be set for NAT;
# the ports below are informational (used for the security-group reminder).
SIGNAL_PORT = 49100  # TCP
MEDIA_PORT = 47998   # UDP


def detect_public_ip() -> str:
    """Best-effort lookup of this server's public IPv4 (for the WebRTC stream).

    Tries the EC2 instance metadata service (IMDSv2) first, then falls back to
    an external echo service.
    """
    try:
        token = subprocess.run(
            ["curl", "-s", "-X", "PUT", "http://169.254.169.254/latest/api/token",
             "-H", "X-aws-ec2-metadata-token-ttl-seconds: 60"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        ip = subprocess.run(
            ["curl", "-s", "-H", f"X-aws-ec2-metadata-token: {token}",
             "http://169.254.169.254/latest/meta-data/public-ipv4"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        if ip:
            return ip
    except Exception:  # noqa: BLE001
        pass
    return subprocess.run(["curl", "-s", "ifconfig.me"], capture_output=True, text=True, timeout=5).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Record teleop demos over WebRTC.")
    parser.add_argument("--profile", default="franka", help="franka (default) or gr1t2.")
    parser.add_argument("--num-demos", type=int, default=5, help="How many demos to record.")
    parser.add_argument("--device", default="keyboard", help="Teleop device (keyboard works over WebRTC).")
    parser.add_argument("--public-ip", default=None, help="Server public IP (default: auto-detect).")
    args = parser.parse_args()
    profile = get_profile(args.profile)

    require_container()
    ensure_container_dirs()

    public_ip = args.public_ip or detect_public_ip()
    if not public_ip:
        return _fail("Could not determine the server public IP. Pass --public-ip <ip>.")

    out_name = f"teleop_{profile.name}.hdf5"
    out_container = f"{CONTAINER_DATA}/{out_name}"
    out_host = DATASETS_DIR / out_name

    # Tell the WebRTC stream its public IP so video reaches a NAT'd client. The
    # signaling/media ports keep their defaults (49100 / 47998), so this single
    # setting (no spaces) is all we pass through to Kit via --kit_args=...
    kit_arg = f"--kit_args=--/exts/omni.kit.livestream.app/primaryStream/publicIp={public_ip}"

    print("=" * 70)
    print("WebRTC teleop is starting. On your laptop:")
    print(f"  1) Security group inbound must allow TCP {SIGNAL_PORT} + UDP {MEDIA_PORT}")
    print(f"     from your IP.")
    print("  2) Open the 'Isaac Sim WebRTC Streaming Client' and connect to:")
    print(f"         {public_ip}")
    print("  3) Drive the arm with the keyboard (W/S/A/D/Q/E move, Z/X/T/G/C/V")
    print("     rotate, K toggles the gripper). Stack the cubes to record a demo.")
    print(f"  Recording {args.num_demos} demo(s) for task '{profile.base_task}'.")
    print("=" * 70)

    try:
        in_container(
            "./isaaclab.sh -p scripts/tools/record_demos.py "
            f"--task {profile.base_task} --teleop_device {args.device} "
            f"--dataset_file {out_container} --num_demos {args.num_demos} "
            f"--livestream 1 {kit_arg}"
        )
    finally:
        # Always try to pull whatever was recorded back to the host, even if you
        # stop early with Ctrl-C.
        print("\nCopying recorded demos back to the host (if any) ...")
        try:
            cp_from_container(out_container, out_host)
            print(f"OK: {out_host}")
            print(f"Next (optional): python3 scripts/02_annotate.py --profile {profile.name}  "
                  f"# after pointing it at this file")
        except Exception:  # noqa: BLE001
            print("[note] no dataset copied out (none recorded yet?).")
    return 0


def _fail(msg: str) -> int:
    print(f"[ERROR] {msg}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
