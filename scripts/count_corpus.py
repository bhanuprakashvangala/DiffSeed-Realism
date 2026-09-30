#!/usr/bin/env python3
"""Count images per class in a downloaded seed corpus.

Checks a local copy of the soybean corpus against the per-class counts reported
in the manuscripts (Broken 1,002; Immature 1,125; Intact 1,201; Skin-damaged
1,127; Spotted 1,058; total 5,513).

Usage:
    python scripts/count_corpus.py data/soybean_seeds
"""
from __future__ import annotations

import sys
from pathlib import Path

EXPECTED = {"Broken soybeans": 1002, "Immature soybeans": 1125, "Intact soybeans": 1201,
            "Skin-damaged soybeans": 1127, "Spotted soybeans": 1058}
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    root = Path(sys.argv[1])
    # descend through single-directory wrappers left by archive extraction
    while True:
        subs = [p for p in root.iterdir() if p.is_dir()]
        if len(subs) == 1 and not any(f.suffix.lower() in EXTS for f in root.iterdir()):
            root = subs[0]
        else:
            break
    counts = {d.name: sum(1 for f in d.rglob("*") if f.suffix.lower() in EXTS)
              for d in sorted(root.iterdir()) if d.is_dir()}
    ok = True
    for name, n in counts.items():
        exp = EXPECTED.get(name)
        flag = "" if exp is None else ("ok" if exp == n else f"expected {exp}")
        ok &= exp is None or exp == n
        print(f"{name:<24}{n:>7}  {flag}")
    print(f"{'total':<24}{sum(counts.values()):>7}  (expected 5513)")
    return 0 if ok and sum(counts.values()) == 5513 else 1


if __name__ == "__main__":
    raise SystemExit(main())
