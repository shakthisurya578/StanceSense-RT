"""Find byte-identical duplicate videos in the Multi-View archive and write the
exclusion list the training/evaluation code must honour.

The archive has 1042 files but 1008 distinct recordings; some copies are filed
under two different people or under both "squat" and a non-squat exercise. See
``resolve_duplicates`` in src/stancesense/datasets/multiview_fitness.py for the
rules. Output: data/processed_multiview/duplicates.json

    python scripts/dedupe_multiview.py
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from stancesense.datasets.multiview_fitness import (  # noqa: E402
    DUPLICATES_FILE, resolve_duplicates, scan_duplicate_recordings,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zip", default=None, help="the dataset archive (auto-detected under data/)")
    ap.add_argument("--processed", default="data/processed_multiview")
    args = ap.parse_args()
    zp = args.zip or (glob.glob("data/A Multi-View Raw Video Dataset*/*.zip") or [None])[0]
    if not zp:
        raise SystemExit("dataset zip not found; pass --zip")
    groups = scan_duplicate_recordings(zp)
    excluded = resolve_duplicates(groups)
    out = dict(source=os.path.basename(zp), n_duplicate_groups=len(groups),
               n_excluded=len(excluded), reasons=dict(Counter(
                   r if not r.startswith("duplicate of") else "extra copy, same person/label/view"
                   for r in excluded.values())),
               groups=groups, excluded=excluded)
    path = os.path.join(args.processed, DUPLICATES_FILE)
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"{len(groups)} duplicate groups, {len(excluded)} clips excluded -> {path}")
    for r, n in out["reasons"].items():
        print(f"  {n:3d}  {r}")


if __name__ == "__main__":
    main()
