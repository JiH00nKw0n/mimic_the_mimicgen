"""AttemptRecord — the single data contract shared by every genaudit module.

One record = one generation attempt (successful or failed). Extractors for any
generation backend (robosuite MimicGen, Isaac Lab Mimic, ...) produce this
schema; everything downstream (binning, sampling, analysis) consumes only this.
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator


@dataclass(frozen=True)
class ObjectDisplacement:
    """Displacement of one task object between the source demo's initial pose
    and the attempted scene's initial pose."""

    object_name: str
    dxy_m: float
    dyaw_rad: float


@dataclass(frozen=True)
class AttemptRecord:
    task: str
    variant: str
    attempt_id: str  # stable reference, e.g. "demo_3@demo_failed.hdf5"
    source_demo_id: int
    success: bool
    episode_length: int
    displacements: tuple[ObjectDisplacement, ...]
    # The three distance definitions of PLAN.md §1.3, always recorded together
    # so definition-robustness checks never require re-extraction.
    d_raw: float  # sum_m ||dxy|| in meters (Phase-0 definition)
    d_pos: float  # (1/M) sum_m ||dxy|| / diag(V_m), in [0, 1]
    d_rot: float  # (1/M_rot) sum_m |wrap_n(dyaw)| / (pi/n_m), in [0, 1]
    extras: dict = field(default_factory=dict)  # factor-axis extension columns

    def __post_init__(self) -> None:
        bad_keys = [key for key in self.extras if not isinstance(key, str)]
        if bad_keys:
            raise TypeError(
                f"extras keys must be strings (JSONL columns), got {bad_keys!r}"
            )

    def to_json_line(self) -> str:
        payload = dataclasses.asdict(self)
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)

    @staticmethod
    def from_json_line(line: str) -> "AttemptRecord":
        payload = json.loads(line)
        displacements = tuple(
            ObjectDisplacement(**item) for item in payload.pop("displacements")
        )
        return AttemptRecord(displacements=displacements, **payload)


def distance_value(record: AttemptRecord, key: str) -> float:
    """Resolve a distance/factor column: schema attribute or extras column.

    The single lookup path every axis consumer (binning, curation, analysis)
    goes through, so a new factor axis only has to write extras columns.
    """
    if key in ("d_raw", "d_pos", "d_rot"):
        return getattr(record, key)
    if key in record.extras:
        return float(record.extras[key])
    raise KeyError(
        f"unknown distance key {key!r}: not a schema field (d_raw/d_pos/d_rot) "
        f"and not in extras (have {sorted(record.extras)})"
    )


def write_jsonl(records: Iterable[AttemptRecord], path: str | Path) -> int:
    """Write records to a JSONL file. Returns the number of records written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.to_json_line() + "\n")
            count += 1
    return count


def read_jsonl(path: str | Path) -> Iterator[AttemptRecord]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield AttemptRecord.from_json_line(line)
