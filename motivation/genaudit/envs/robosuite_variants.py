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
    "hammer_cleanup": "HammerCleanup",
    "coffee_preparation": "CoffeePreparation",
}

# motivation_new: parent (module, class) for the N0/N1/N2 isotropic ladder.
# Every task's three N-variants share one parent (its D1 class carries the full
# reset machinery); only our bounds override differs.
NEW_VARIANT_PARENTS: dict[str, tuple[str, str]] = {
    "square": ("mimicgen.envs.robosuite.nut_assembly", "Square_D1"),
    "threading": ("mimicgen.envs.robosuite.threading", "Threading_D1"),
    "coffee": ("mimicgen.envs.robosuite.coffee", "Coffee_D1"),
    "three_piece_assembly": ("mimicgen.envs.robosuite.three_piece_assembly", "ThreePieceAssembly_D1"),
    "stack": ("mimicgen.envs.robosuite.stack", "Stack_D1"),
    "stack_three": ("mimicgen.envs.robosuite.stack", "StackThree_D1"),
    "mug_cleanup": ("mimicgen.envs.robosuite.mug_cleanup", "MugCleanup_D1"),
    "hammer_cleanup": ("mimicgen.envs.robosuite.hammer_cleanup", "HammerCleanup_D1"),
    "coffee_preparation": ("mimicgen.envs.robosuite.coffee", "CoffeePreparation_D1"),
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


# Task-specific placement-dict quirks (only for tasks whose env placement code
# differs from the registry format). Discovered by the validation smoke.
#   * Square's placement uses object keys "nut"/"peg", not "square_nut"/"square_peg".
#   * Hammer's placement reads a per-object "rotation_axis" field.
PLACEMENT_KEY_MAP: dict[str, dict[str, str]] = {
    "square": {"square_nut": "nut", "square_peg": "peg"},
}
PLACEMENT_EXTRA: dict[str, dict[str, dict]] = {
    "hammer_cleanup": {"hammer": {"rotation_axis": "y"}},
}


def _placement_bounds_for(task, variant_bounds, reference):
    """to_mimicgen_bounds + per-task key remap / extra fields."""
    import numpy as np

    kmap = PLACEMENT_KEY_MAP.get(task, {})
    extra = PLACEMENT_EXTRA.get(task, {})
    out = {}
    for name, bounds in variant_bounds.items():
        entry = {
            "x": bounds.x, "y": bounds.y, "z_rot": bounds.z_rot,
            "reference": np.array(reference),
        }
        entry.update(extra.get(name, {}))
        out[kmap.get(name, name)] = entry
    return out


def register_new_variants() -> dict[str, type]:
    """Build and register the motivation_new N0/N1/N2 isotropic ladder for all
    9 tasks (bounds from genaudit.envs.bounds_new). Idempotent."""
    import importlib

    from genaudit.envs.bounds import REFERENCE_OFFSETS
    from genaudit.envs.bounds_new import NEW_BOUNDS

    try:
        importlib.import_module("mimicgen")
    except ImportError as error:  # pragma: no cover - env-dependent
        raise ImportError(
            "mimicgen is required to register env variants — run on the server"
        ) from error

    created: dict[str, type] = {}
    for task, (module_name, parent_name) in NEW_VARIANT_PARENTS.items():
        module = importlib.import_module(module_name)
        parent = getattr(module, parent_name)
        for variant in ("N0", "N1", "N2"):
            class_name = variant_class_name(task, variant)
            if hasattr(module, class_name):
                created[class_name] = getattr(module, class_name)
                continue
            bounds = _placement_bounds_for(
                task, NEW_BOUNDS[task][variant], REFERENCE_OFFSETS[task]
            )

            def _make_bounds_method(frozen_bounds):
                def _get_initial_placement_bounds(self):
                    return frozen_bounds

                return _get_initial_placement_bounds

            variant_class = type(
                class_name,
                (parent,),
                {
                    "_get_initial_placement_bounds": _make_bounds_method(bounds),
                    "__doc__": f"{class_name}: motivation_new isotropic rotation-free variant.",
                    "__module__": module_name,
                },
            )
            setattr(module, class_name, variant_class)
            created[class_name] = variant_class
    return created
