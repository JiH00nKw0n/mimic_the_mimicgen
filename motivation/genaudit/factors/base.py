"""FactorAxis protocol — the extension point for new diversity axes.

An axis contributes (a) named scalar columns per attempt (stored in
`AttemptRecord.extras` by extractors) and (b) a primary scalar used for
binning/stratification. `initial_condition` is the first implementation;
proprioception / skill-action / motion-action axes plug in here later without
touching binning, sampling, or analysis code.
"""
from __future__ import annotations

from typing import Mapping, Protocol, runtime_checkable

from genaudit.records.schema import AttemptRecord, distance_value


@runtime_checkable
class FactorAxis(Protocol):
    name: str

    def columns(self, record: AttemptRecord) -> Mapping[str, float]:
        """Named scalar columns this axis contributes for one attempt."""
        ...

    def primary_value(self, record: AttemptRecord) -> float:
        """The scalar used for binning/stratification along this axis."""
        ...


class InitialConditionAxis:
    """Reads the distances that extractors already stored on the record.

    primary_value goes through records.distance_value — the same resolver the
    CLI/analysis use — so a future axis whose columns live in extras plugs into
    binning/curation without touching them.
    """

    name = "initial_condition"

    def __init__(self, primary_key: str = "d_pos") -> None:
        self.primary_key = primary_key

    def columns(self, record: AttemptRecord) -> Mapping[str, float]:
        return {"d_raw": record.d_raw, "d_pos": record.d_pos, "d_rot": record.d_rot}

    def primary_value(self, record: AttemptRecord) -> float:
        return distance_value(record, self.primary_key)
