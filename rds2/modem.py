from __future__ import annotations

import math
import numpy as np
from scipy.signal import resample_poly


def differential_encode(bits: np.ndarray) -> np.ndarray:
    """Differential encoding for RDS: 1 -> phase invert, 0 -> no change. Output +/-1 as float."""
    out = np.empty_like(bits, dtype=np.int8)
    phase = 1
    for i, b in enumerate(bits):
        if b:
            phase = -phase
        out[i] = phase
    return out.astype(np.float64)


def raised_cosine(num_taps: int, beta: float, sps: float) -> np.ndarray:
    """Raised cosine pulse shape filter (FIR) for BPSK shaping."""
    t = (np.arange(num_taps) - (num_taps - 1) / 2.0) / sps
    h = np.zeros_like(t)
    for i, ti in enumerate(t):
        if abs(1 - (2 * beta * ti) ** 2) < 1e-8:
            h[i] = math.pi / 4 * np.sinc(1 / (2 * beta))
        else:
            h[i] = np.sinc(ti) * (np.cos(math.pi * beta * ti) / (1 - (2 * beta * ti) ** 2))
    h = h / np.sum(h)
    return h


def bpsk_subcarrier(
    bits: np.ndarray,
    fs: float,
    subcarrier_hz: float,
    bitrate: float,
    beta: float = 0.5,
    span_symbols: int = 6,
) -> np.ndarray:
    """Generate BPSK with differential encoding and raised-cosine shaping, mixed to subcarrier."""
    sps = fs / bitrate
    if sps < 4:
        raise ValueError("Sampling rate too low for RDS/RDS2")

    symbols = differential_encode(bits)

    up_factor = int(round(sps))
    if abs(sps - up_factor) > 1e-6:
        base = np.repeat(symbols, int(math.ceil(sps)))
        sps_eff = int(math.ceil(sps))
    else:
        base = np.zeros(len(symbols) * up_factor)
        base[::up_factor] = symbols
        sps_eff = up_factor

    num_taps = max(41, int(span_symbols * sps_eff) | 1)
    h = raised_cosine(num_taps=num_taps, beta=beta, sps=float(sps_eff))
    shaped = np.convolve(base, h, mode='same')

    if sps_eff != sps:
        shaped = resample_poly(shaped, up=int(fs), down=int(bitrate * sps_eff))

    t = np.arange(len(shaped)) / fs
    carrier = np.cos(2 * np.pi * subcarrier_hz * t)
    return shaped * carrier