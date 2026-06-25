"""Re-expose `warp.torch` for cuRobo 0.7.7 on warp 1.14.

cuRobo 0.7.7 (the pinned NVIDIA build) calls `wp.torch.device_from_torch(...)` inside its
collision-checker construction. warp 1.14 moved the torch interop to `warp._src.torch` and
dropped the public `warp.torch` namespace, so that call raises
`AttributeError: module 'warp' has no attribute 'torch'` at MotionGen build time.

Importing this module (before any cuRobo planner is built) restores `warp.torch` /
`wp.torch` by aliasing the relocated `warp._src.torch`. No edit to the shared cuRobo or
warp install is needed — we only register the alias in this process.
"""

from __future__ import annotations

import sys


def install() -> None:
    import warp as wp

    if getattr(wp, "torch", None) is not None and "warp.torch" in sys.modules:
        return
    try:
        import warp._src.torch as _wp_torch  # warp >= ~1.14
    except ModuleNotFoundError:
        try:
            import warp.torch as _wp_torch  # older warp: already public, nothing to do
        except ModuleNotFoundError:
            return
    sys.modules.setdefault("warp.torch", _wp_torch)
    wp.torch = _wp_torch


install()
