"""
Write valid tiny JPEG placeholders into dataset/*/sample1.jpg and sample2.jpg
(stdlib only — no OpenCV required). For 224×224 labeled tiles, run:
    python scripts/bootstrap_dataset_samples.py

From project root:
    python scripts/write_placeholder_jpegs.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dataset_utils import write_sample_placeholders  # noqa: E402


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    write_sample_placeholders(root)
    print("Wrote dataset/cheating/sample{1,2}.jpg and dataset/not_cheating/sample{1,2}.jpg")


if __name__ == "__main__":
    main()
