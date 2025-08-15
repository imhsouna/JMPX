from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np
try:
    import sounddevice as sd
except Exception:
    sd = None  # type: ignore
import soundfile as sf
from scipy.signal import resample_poly

from .config import db_to_linear


def read_audio_file(path: str, target_fs: int) -> Tuple[np.ndarray, int]:
    data, fs = sf.read(path, always_2d=True)
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    if fs != target_fs:
        left = resample_poly(data[:, 0], up=target_fs, down=fs)
        right = resample_poly(data[:, 1], up=target_fs, down=fs)
        data = np.stack([left, right], axis=1)
        fs = target_fs
    return data.astype(np.float32), fs


def generate_tone(duration_s: float, fs: int, freq_hz: float = 1000.0, level_db: float = -12.0) -> np.ndarray:
    t = np.arange(int(duration_s * fs)) / fs
    amp = db_to_linear(level_db)
    left = amp * np.sin(2 * np.pi * freq_hz * t)
    right = left.copy()
    return np.stack([left, right], axis=1).astype(np.float32)


def find_device_index_by_name(name_query: str, is_output: Optional[bool] = None) -> Optional[int]:
    """Find a device index whose name contains the given query (case-insensitive).
    If is_output is True, restrict to output-capable devices. If False, input-capable. If None, any.
    Prefer exact match if multiple, otherwise first partial match.
    """
    if sd is None:
        return None
    devices = sd.query_devices()
    name_lc = name_query.lower()
    candidates: List[Tuple[int, dict]] = []
    for idx, dev in enumerate(devices):
        if is_output is True and dev.get("max_output_channels", 0) <= 0:
            continue
        if is_output is False and dev.get("max_input_channels", 0) <= 0:
            continue
        if name_lc in dev.get("name", "").lower():
            candidates.append((idx, dev))
    if not candidates:
        return None
    for idx, dev in candidates:
        if dev.get("name", "").lower() == name_lc:
            return idx
    return candidates[0][0]