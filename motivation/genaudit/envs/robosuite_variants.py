"""Create and register the E-series robosuite env variants from the registry.

The bounds live in `genaudit.envs.bounds` (pure data, tested); this module
turns them into mimicgen env subclasses at runtime. robosuite registers every
env subclass at class-definition time, so importing this module and calling
`register_custom_variants()` before `generate_dataset.py` runs is all the
integration needed — mimicgen source stays untouched.

Parent classes are chosen so each E-variant inherits its task's special
machinery (camera swaps, collision-aware resets) from the closest public
variant.
"""
from __future__ import annotations

from typing import Mapping

from genaudit.envs.bounds import BOUNDS, REFERENCE_OFFSETS, PlacementBounds

# (task, variant) -> (parent module, parent class name). Parents are D1-level
# classes: they carry the task's extra reset machinery while our bounds
# override widens the regions.
CUSTOM_VARIANT_PARENTS: dict[tuple[str, str], tuple[str, str]] = {
    ("threading", "D2E"): ("mimicgen.envs.robosuite.threading", "Threading_D1"),
    ("coffee", "D1E"): ("mimicgen.envs.robosuite.coffee", "Coffee_D1"),
    ("coffee", "D2E"): ("mimicgen.envs.robosuite.coffee", "Coffee_D1"),
    ("stack", "D2E"): ("mimicgen.envs.robosuite.stack", "Stack_D1"),
    ("stack_three", "D2E"): ("mimicgen.envs.robosuite.stack", "StackThree_D1"),
    ("mug_cleanup", "D1E"): ("mimicgen.envs.robosuite.mug_cleanup", "MugCleanup_D1"),
    ("mug_cleanup", "D2E"): ("mimicgen.envs.robosuite.mug_cleanup", "MugCleanup_D1"),
}

_CLASS_NAME_BY_TASK = {
    "threading": "Threading",
    "coffee": "Coffee",
    "stack": "Stack",
    "stack_three": "StackThree",
    "mug_cleanup": "MugCleanup",
    "square": "Square",
    "three_piece_assembly": "ThreePieceAssembly",
}


def variant_class_name(task: str, variant: str) -> str:
    """E.g. ("threading", "D2E") -> "Threading_D2E" — the --task_name value.

    Works for public variants too (Square_D2 etc.); only E-suffixed variants
    need registration via register_custom_variants().
    """
    if task not in _CLASS_NAME_BY_TASK:
        raise KeyError(
            f"unknown task {task!r}; known: {sorted(_CLASS_NAME_BY_TASK)}"
        )
    return f"{_CLASS_NAME_BY_TASK[task]}_{variant}"


def to_mimicgen_bounds(
    variant_bounds: Mapping[str, PlacementBounds], reference: tuple[float, float, float]
):
    """Convert registry bounds to the dict shape mimicgen envs return from
    `_get_initial_placement_bounds()`."""
    import numpy as np

    return {
        name: {
            "x": bounds.x,
            "y": bounds.y,
            "z_rot": bounds.z_rot,
            "reference": np.array(reference),
        }
        for name, bounds in variant_bounds.items()
    }


def register_custom_variants() -> dict[str, type]:
    """Build and register every E-series env class. Idempotent.

    Raises ImportError with install guidance when mimicgen is unavailable.
    """
    import importlib

    try:
        importlib.import_module("mimicgen")
    except ImportError as error:  # pragma: no cover - env-dependent
        raise ImportError(
            "mimicgen is required to register env variants — run this on the "
            "generation server (robosuite_mimicgen venv), not the laptop"
        ) from error

    created: dict[str, type] = {}
    for (task, variant), (module_name, parent_name) in CUSTOM_VARIANT_PARENTS.items():
        module = importlib.import_module(module_name)
        parent = getattr(module, parent_name)
        class_name = variant_class_name(task, variant)
        if hasattr(module, class_name):  # already registered in this process
            created[class_name] = getattr(module, class_name)
            continue
        bounds = to_mimicgen_bounds(BOUNDS[task][variant], REFERENCE_OFFSETS[task])

        def _make_bounds_method(frozen_bounds):
            def _get_initial_placement_bounds(self):
                return frozen_bounds

            return _get_initial_placement_bounds

        variant_class = type(
            class_name,
            (parent,),
            {
                "_get_initial_placement_bounds": _make_bounds_method(bounds),
                "__doc__": (
                    f"{class_name}: E-series strict-superset expansion variant "
                    f"(genaudit registry; see motivation/TASKS.md §3)."
                ),
                "__module__": module_name,
            },
        )
        # Attach to the mimicgen module so repeated registration is a no-op
        # and robosuite's name-based env lookup can resolve the class.
        setattr(module, class_name, variant_class)
        created[class_name] = variant_class
    return created
