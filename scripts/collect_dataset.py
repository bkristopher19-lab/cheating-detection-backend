"""
Collect labeled webcam images for cheating / not-cheating model training.

Requirements: OpenCV with GUI support for ``cv2.imshow``. This repo lists
``opencv-python-headless``; for the live preview window, also install
``opencv-python`` in your venv (or any OpenCV build that includes HighGUI).

Usage (from project root):
    python scripts/collect_dataset.py

Controls:
    c  — save current frame to dataset/cheating/   (resized 224×224)
    n  — save current frame to dataset/not_cheating/
    q  — quit
"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

# Project root = parent of scripts/
ROOT = Path(__file__).resolve().parent.parent
CHEATING_DIR = ROOT / "dataset" / "cheating"
NOT_CHEATING_DIR = ROOT / "dataset" / "not_cheating"
SAVE_SIZE = (224, 224)
WINDOW_TITLE = "Dataset collector | c=cheating | n=not cheating | q=quit"


def ensure_dirs() -> None:
    """Create dataset folders if they are missing."""
    CHEATING_DIR.mkdir(parents=True, exist_ok=True)
    NOT_CHEATING_DIR.mkdir(parents=True, exist_ok=True)


def unique_filename(prefix: str) -> str:
    """Return a unique filename using a timestamp (safe for Windows/Linux/macOS)."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ms = int(time.time() * 1000) % 1000
    return f"{prefix}_{ts}_{ms:03d}.jpg"


def save_frame(frame_bgr, dest_dir: Path, prefix: str) -> Path:
    """Resize frame to 224×224 and save as JPEG."""
    resized = cv2.resize(frame_bgr, SAVE_SIZE, interpolation=cv2.INTER_AREA)
    path = dest_dir / unique_filename(prefix)
    cv2.imwrite(str(path), resized, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return path


def open_webcam():
    """Open the default camera (Windows may need CAP_DSHOW for some drivers)."""
    if sys.platform == "win32":
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if cap.isOpened():
            return cap
    cap = cv2.VideoCapture(0)
    return cap if cap.isOpened() else None


def main() -> None:
    ensure_dirs()

    cap = open_webcam()
    if cap is None:
        print("Error: could not open webcam (try another camera index or check permissions).")
        print("Tip: on servers without a camera, use images from another machine or the sample placeholders.")
        return

    print(__doc__)
    print(f"Saving cheating images to:    {CHEATING_DIR}")
    print(f"Saving not_cheating images to: {NOT_CHEATING_DIR}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Warning: failed to read frame; retrying...")
                continue

            # Preview can stay full resolution; saved files are always 224×224
            preview = frame.copy()
            cv2.putText(
                preview,
                "c=cheating  n=not cheating  q=quit",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 200, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(WINDOW_TITLE, preview)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                path = save_frame(frame, CHEATING_DIR, "cheating")
                print(f"Saved (cheating): {path.name}")
            elif key == ord("n"):
                path = save_frame(frame, NOT_CHEATING_DIR, "not_cheating")
                print(f"Saved (not_cheating): {path.name}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Done.")


if __name__ == "__main__":
    main()
