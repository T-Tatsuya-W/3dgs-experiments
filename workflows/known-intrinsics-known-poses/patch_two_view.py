#!/usr/bin/env python3
"""Keep both images in training for a two-view gsplat v1.5.3 dataset."""
from pathlib import Path

path = Path("third_party/gsplat/examples/datasets/colmap.py")
if not path.exists():
    raise SystemExit("Run from project root after checking out gsplat v1.5.3.")
source = path.read_text(encoding="utf8")
old = '        if split == "train":\n            self.indices = indices[indices % self.parser.test_every != 0]'
new = '        if split == "train":\n            self.indices = indices if len(indices) <= 2 else indices[indices % self.parser.test_every != 0]'
if old in source:
    path.write_text(source.replace(old, new), encoding="utf8")
    print("Patched two-image training split; both images used for optimisation.")
elif new in source:
    print("Two-image split already patched.")
else:
    raise SystemExit("Unexpected upstream trainer source; use exact gsplat v1.5.3 checkout.")
