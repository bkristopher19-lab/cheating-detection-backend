"""
Create sample1.jpg and sample2.jpg in dataset/cheating and dataset/not_cheating.

Preferred: 224×224 color tiles (needs NumPy + OpenCV).
Fallback: tiny valid JPEG bytes (stdlib only via dataset_utils).

From project root:
    python scripts/bootstrap_dataset_samples.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


def main() -> None:
    try:
        import cv2
        import numpy as np
    except ImportError:
        from dataset_utils import write_sample_placeholders

        write_sample_placeholders(ROOT)
        print("NumPy/OpenCV not found — wrote minimal JPEG placeholders (1×1).")
        print("Install requirements.txt and re-run for 224×224 samples.")
        return

    specs = [
        ("cheating", [(80, 40, 40), (120, 50, 50)]),
        ("not_cheating", [(40, 100, 60), (50, 120, 70)]),
    ]
    for subfolder, colors in specs:
        folder = ROOT / "dataset" / subfolder
        folder.mkdir(parents=True, exist_ok=True)
        for i, bgr in enumerate(colors, start=1):
            img = np.full((224, 224, 3), bgr, dtype=np.uint8)
            cv2.putText(
                img,
                f"sample{i}",
                (60, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            out = folder / f"sample{i}.jpg"
            cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            print(f"Wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
