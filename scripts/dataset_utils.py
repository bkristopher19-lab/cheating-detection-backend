"""
Shared helpers for the cheating-detection image dataset (minimal JPEG, paths).
"""

from __future__ import annotations

from pathlib import Path

# Minimal valid JFIF JPEG (1×1 grey), 134-byte body + FFD9 EOI. Quant table = 64 bytes after FF DB 00 43 00.
_MINI_JPEG_HEX = (
    "ffd8ffe000104a46494600010101004800480000"
    "ffdb004300"
    + ("ff" * 64)
    + "c0000b080001000101011100ffc4001410010000000000000000000000000000"
    + "ffda0008010100013f10"
    + "ffd9"
)

MINI_JPEG_BYTES = bytes.fromhex(_MINI_JPEG_HEX)


def write_sample_placeholders(root: Path) -> None:
    """Write sample1.jpg and sample2.jpg (minimal JPEG) under dataset/cheating and dataset/not_cheating."""
    for sub in ("cheating", "not_cheating"):
        d = root / "dataset" / sub
        d.mkdir(parents=True, exist_ok=True)
        for name in ("sample1.jpg", "sample2.jpg"):
            (d / name).write_bytes(MINI_JPEG_BYTES)
