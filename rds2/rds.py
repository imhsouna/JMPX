from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np

from .config import (
    RdsConfig,
    RDS_CRC_POLY,
    RDS_OFFSET_A,
    RDS_OFFSET_B,
    RDS_OFFSET_C,
    RDS_OFFSET_D,
    RDS2_LOGO_MAGIC,
)


def _rds_crc10(word16: int) -> int:
    """Compute 10-bit CRC checkword for a 16-bit block using RDS polynomial."""
    reg = 0
    data = word16 << 10
    mask = 1 << 25  # process 26 bits total
    for _ in range(26):
        bit = 1 if (data & mask) else 0
        top = 1 if (reg & (1 << 10)) else 0
        reg = ((reg << 1) & 0x7FF) | bit
        if top:
            reg ^= RDS_CRC_POLY
        mask >>= 1
    return reg & 0x3FF


def _rds_block(word16: int, offset_word: int) -> tuple[int, int]:
    cw = _rds_crc10(word16) ^ offset_word
    return word16 & 0xFFFF, cw & 0x3FF


def _pack_bits_from_blocks(blocks: List[Tuple[int, int]]) -> np.ndarray:
    """Convert (word16, cw10) pairs into a flat bit array (MSB-first per word)."""
    bits: List[int] = []
    for word, cw in blocks:
        for i in range(15, -1, -1):
            bits.append((word >> i) & 1)
        for i in range(9, -1, -1):
            bits.append((cw >> i) & 1)
    return np.array(bits, dtype=np.uint8)


def build_group_0a(cfg: RdsConfig, ps_pair_index: int) -> np.ndarray:
    """Group 0A: Basic tuning and switching information + PS name segments.
    ps_pair_index: 0..3 selects which 2-char pair of the 8-char PS to send.
    """
    ps = (cfg.program_service_name or "").ljust(8)[:8]
    segment_ch = ps_pair_index & 0x3
    c1 = ps[segment_ch * 2]
    c2 = ps[segment_ch * 2 + 1]

    block_a = cfg.pi_code & 0xFFFF

    group_type = 0  # 0A
    version_a = 0
    tp = 1 if cfg.tp else 0
    pty = cfg.pty & 0x1F

    block_b = 0
    block_b |= (tp & 1) << 10
    block_b |= (pty & 0x1F) << 5
    block_b |= (group_type & 0xF) << 1
    block_b |= (version_a & 0x1)
    block_b |= (segment_ch & 0x3)

    block_c = 0x0000

    block_d = ((ord(c1) & 0xFF) << 8) | (ord(c2) & 0xFF)

    blocks = [
        _rds_block(block_a, RDS_OFFSET_A),
        _rds_block(block_b, RDS_OFFSET_B),
        _rds_block(block_c, RDS_OFFSET_C),
        _rds_block(block_d, RDS_OFFSET_D),
    ]
    return _pack_bits_from_blocks(blocks)


def build_group_2a(cfg: RdsConfig, rt_pair_index: int) -> np.ndarray:
    """Group 2A: Radiotext, 64 chars in 2-char pairs across 16 blocks; rt_pair_index 0..15"""
    text = (cfg.radiotext or "").ljust(64)[:64]
    pair_idx = rt_pair_index & 0x0F
    c1 = text[pair_idx * 4 + 0]
    c2 = text[pair_idx * 4 + 1]
    c3 = text[pair_idx * 4 + 2]
    c4 = text[pair_idx * 4 + 3]

    block_a = cfg.pi_code & 0xFFFF

    group_type = 2
    version_a = 0
    tp = 1 if cfg.tp else 0
    pty = cfg.pty & 0x1F

    block_b = 0
    block_b |= (tp & 1) << 10
    block_b |= (pty & 0x1F) << 5
    block_b |= (group_type & 0xF) << 1
    block_b |= (version_a & 0x1)
    block_b |= (pair_idx & 0x0F)

    block_c = ((ord(c1) & 0xFF) << 8) | (ord(c2) & 0xFF)
    block_d = ((ord(c3) & 0xFF) << 8) | (ord(c4) & 0xFF)

    blocks = [
        _rds_block(block_a, RDS_OFFSET_A),
        _rds_block(block_b, RDS_OFFSET_B),
        _rds_block(block_c, RDS_OFFSET_C),
        _rds_block(block_d, RDS_OFFSET_D),
    ]
    return _pack_bits_from_blocks(blocks)


class RdsBitstreamGenerator:
    """Generate a continuous RDS bitstream (0/1) by cycling groups 0A and 2A.

    Optionally interleave a simple framed monochrome logo payload for RDS2 sidebands.
    """

    def __init__(self, cfg: RdsConfig):
        self.cfg = cfg
        self.ps_index = 0
        self.rt_index = 0
        self.logo_frame: Optional[np.ndarray] = None
        self.logo_idx = 0

    def set_logo_bits(self, bits: Optional[np.ndarray]) -> None:
        self.logo_frame = bits
        self.logo_idx = 0

    def _next_logo_chunk(self, max_bits: int) -> Optional[np.ndarray]:
        if self.logo_frame is None or len(self.logo_frame) == 0:
            return None
        n = min(max_bits, len(self.logo_frame) - self.logo_idx)
        if n <= 0:
            self.logo_idx = 0
            n = min(max_bits, len(self.logo_frame))
        chunk = self.logo_frame[self.logo_idx : self.logo_idx + n]
        self.logo_idx += n
        return chunk

    def next_group_bits(self) -> np.ndarray:
        # Insert small logo chunk periodically to keep presence in RDS2 sidebands
        if (self.ps_index % 5) == 0 and self.logo_frame is not None:
            chunk = self._next_logo_chunk(104)  # approx group size
            if chunk is not None:
                return chunk

        if (self.ps_index % 3) != 2:
            bits = build_group_0a(self.cfg, self.ps_index & 0x3)
            self.ps_index = (self.ps_index + 1) % 4
            return bits
        else:
            bits = build_group_2a(self.cfg, self.rt_index & 0x0F)
            self.rt_index = (self.rt_index + 1) % 16
            return bits

    def generate_bits(self, total_bits: int) -> np.ndarray:
        out = np.empty(total_bits, dtype=np.uint8)
        filled = 0
        while filled < total_bits:
            group = self.next_group_bits()
            n = min(len(group), total_bits - filled)
            out[filled : filled + n] = group[:n]
            filled += n
        return out