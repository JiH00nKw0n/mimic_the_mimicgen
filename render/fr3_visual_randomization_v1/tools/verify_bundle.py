#!/usr/bin/env python3
"""Verify the handoff bundle against MANIFEST.sha256."""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = root / "MANIFEST.sha256"
    if not manifest.is_file():
        raise SystemExit("MANIFEST.sha256 is missing")
    checked = 0
    failures = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split("  ", 1)
        path = root / relative
        if not path.is_file():
            failures.append(f"missing: {relative}")
        else:
            observed = sha256(path)
            if observed != expected:
                failures.append(f"checksum: {relative} expected={expected} observed={observed}")
        checked += 1
    if failures:
        print("FAIL")
        print("\n".join(failures))
        raise SystemExit(1)
    print(f"PASS: {checked} files, 0 mismatches")


if __name__ == "__main__":
    main()
