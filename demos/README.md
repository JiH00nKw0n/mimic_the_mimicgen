# Demos: human seed vs MimicGen synthetic

MimicGen starts from a handful of human teleoperation demos and multiplies them
into ~1000 synthetic demos. Each pair below shows an original **human** demo
next to a **synthetic** demo MimicGen generated from it (replayed in simulation
and rendered headless to MP4, then converted to GIF for display).

## Franka — cube stacking (single arm)

10 human demos → 1000 synthetic (~37% generation success).

| Human seed demo | MimicGen synthetic |
|:---:|:---:|
| ![franka human demo](franka_human_0.gif) | ![franka synthetic demo](franka_synthetic_0.gif) |

Full-quality MP4s — human: [0](franka_human_0.mp4) · [1](franka_human_1.mp4) — synthetic: [0](franka_synthetic_0.mp4) · [1](franka_synthetic_1.mp4)

## GR1T2 — pick & place (bimanual humanoid)

Pre-annotated human demos → 1000 synthetic (~87% generation success). Left arm
picks, right arm places.

| Human seed demo | MimicGen synthetic |
|:---:|:---:|
| ![gr1t2 human demo](gr1t2_human_0.gif) | ![gr1t2 synthetic demo](gr1t2_synthetic_0.gif) |

Full-quality MP4s — human: [0](gr1t2_human_0.mp4) · [1](gr1t2_human_1.mp4) — synthetic: [0](gr1t2_synthetic_0.mp4) · [1](gr1t2_synthetic_1.mp4)
