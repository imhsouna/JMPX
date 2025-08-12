from __future__ import annotations

import numpy as np
from scipy.signal import firwin, lfilter

from .config import (
    PILOT_HZ,
    STEREO_SUBCARRIER_HZ,
    RDS0_HZ,
    RDS2_SUBCARRIER_HZ,
    DEFAULT_PILOT_LEVEL,
    DEFAULT_RDS_LEVEL,
    DEFAULT_RDS2_LEVEL,
    RDS_BITRATE,
)
from .modem import bpsk_subcarrier
from .config import clamp_audio


def lowpass_stereo(audio: np.ndarray, fs: float, cutoff_hz: float = 15000.0) -> np.ndarray:
    numtaps = 513
    fir = firwin(numtaps, cutoff=cutoff_hz, fs=fs)
    return lfilter(fir, [1.0], audio, axis=0)


def make_mpx(
    left: np.ndarray,
    right: np.ndarray,
    fs: float,
    pilot_level: float = DEFAULT_PILOT_LEVEL,
    rds_level: float = DEFAULT_RDS_LEVEL,
    rds2_level: float = DEFAULT_RDS2_LEVEL,
    rds_bits: np.ndarray | None = None,
    enable_rds2: bool = False,
) -> np.ndarray:
    assert left.shape == right.shape
    num_samples = left.shape[0]

    stereo = np.stack([left, right], axis=1)
    stereo = lowpass_stereo(stereo, fs)
    lpr = np.mean(stereo, axis=1)  # L+R
    lmr = stereo[:, 0] - stereo[:, 1]  # L-R

    t = np.arange(num_samples) / fs

    pilot = pilot_level * np.sin(2 * np.pi * PILOT_HZ * t)

    stereo_sub = np.cos(2 * np.pi * STEREO_SUBCARRIER_HZ * t)
    dsb = lmr * stereo_sub

    rds = np.zeros(num_samples)
    if rds_bits is not None and len(rds_bits) > 0:
        rds_wave = bpsk_subcarrier(rds_bits, fs=fs, subcarrier_hz=RDS0_HZ, bitrate=RDS_BITRATE)
        rds[: len(rds_wave)] += rds_level * rds_wave[:num_samples]

    if enable_rds2 and rds_bits is not None and len(rds_bits) > 0:
        for sc in RDS2_SUBCARRIER_HZ:
            rds2_wave = bpsk_subcarrier(rds_bits, fs=fs, subcarrier_hz=sc, bitrate=RDS_BITRATE)
            rds[: len(rds2_wave)] += rds2_level * rds2_wave[:num_samples]

    mpx = lpr + pilot + dsb + rds
    return clamp_audio(mpx.astype(np.float32))