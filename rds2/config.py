from __future__ import annotations

import numpy as np
from dataclasses import dataclass

# Base subcarrier and rates
PILOT_HZ: float = 19000.0
STEREO_SUBCARRIER_HZ: float = 38000.0
RDS0_HZ: float = 57000.0
# Experimental RDS2 additional carriers (as per ETSI TS 103 634): 66.5/76/85.5 kHz
RDS2_SUBCARRIER_HZ = [66500.0, 76000.0, 85500.0]
RDS_BITRATE: float = 1187.5

# Levels (ratios)
DEFAULT_PILOT_LEVEL: float = 0.08
DEFAULT_RDS_LEVEL: float = 0.03
DEFAULT_RDS2_LEVEL: float = 0.01

# RDS2 experimental logo framing
RDS2_LOGO_MAX_W: int = 64
RDS2_LOGO_MAX_H: int = 32
RDS2_LOGO_MAGIC: int = 0xA7

# RDS CRC generator polynomial g(x) = x^10 + x^8 + x^7 + x^5 + x^4 + x^3 + 1
RDS_CRC_POLY: int = 0x5B9

# Offset words for blocks A, B, C, D (10-bit)
RDS_OFFSET_A: int = 0x0FC
RDS_OFFSET_B: int = 0x198
RDS_OFFSET_C: int = 0x168
RDS_OFFSET_D: int = 0x1B4


@dataclass
class RdsConfig:
    pi_code: int
    pty: int = 0
    tp: int = 0
    program_service_name: str = ""
    radiotext: str = ""


def db_to_linear(db_value: float) -> float:
    return 10.0 ** (db_value / 20.0)


def clamp_audio(signal: np.ndarray, peak: float = 0.999) -> np.ndarray:
    return np.clip(signal, -peak, peak)