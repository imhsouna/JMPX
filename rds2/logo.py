from __future__ import annotations

import numpy as np
from PIL import Image

from .config import (
    RDS2_LOGO_MAX_W,
    RDS2_LOGO_MAX_H,
    RDS2_LOGO_MAGIC,
)


def load_logo_bits(path: str) -> np.ndarray:
    """Load an image and pack as a simple framed monochrome bitstream for experimental RDS2.

    Frame format (repeating):
    - 8 bits magic (0xA7)
    - 7 bits width (1..64)
    - 6 bits height (1..32)
    - 3 bits reserved (0)
    - width*height bits, row-major, 1=white, 0=black
    - 16 bits simple checksum (sum of payload bytes & 0xFFFF)
    """
    img = Image.open(path).convert('L')
    w = min(RDS2_LOGO_MAX_W, max(1, img.width))
    h = min(RDS2_LOGO_MAX_H, max(1, img.height))
    if img.width != w or img.height != h:
        img = img.resize((w, h), Image.LANCZOS)
    arr = np.array(img)
    thr = float(arr.mean())
    bits = (arr >= thr).astype(np.uint8)

    header: list[int] = []

    def put(val: int, nbits: int) -> None:
        for i in range(nbits - 1, -1, -1):
            header.append((val >> i) & 1)

    put(RDS2_LOGO_MAGIC, 8)
    put(w, 7)
    put(h, 6)
    put(0, 3)

    payload_bits = bits.flatten().tolist()

    payload_bytes: list[int] = []
    acc = 0
    cnt = 0
    for b in payload_bits:
        acc = (acc << 1) | b
        cnt += 1
        if cnt == 8:
            payload_bytes.append(acc)
            acc = 0
            cnt = 0
    if cnt != 0:
        acc <<= (8 - cnt)
        payload_bytes.append(acc)

    checksum = sum(payload_bytes) & 0xFFFF
    footer: list[int] = []
    put(checksum, 16)

    all_bits = np.array(header + payload_bits + footer, dtype=np.uint8)
    return all_bits