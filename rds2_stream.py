#!/usr/bin/env python3
import math
import sys
import time
import queue
from dataclasses import dataclass
from typing import List, Optional, Tuple

import click
import numpy as np
import sounddevice as sd
import soundfile as sf
# Optional SciPy (Windows Store Python sometimes hangs on import). We provide fallbacks.
try:
    from scipy.signal import resample_poly as sp_resample_poly, firwin as sp_firwin, lfilter as sp_lfilter
except Exception:  # ImportError or other runtime issues
    sp_resample_poly = None
    sp_firwin = None
    sp_lfilter = None
from PIL import Image


def list_devices(verbose: bool = False) -> str:
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()
    lines = []
    if not verbose:
        lines.append("Index | Name | Max output channels")
        for idx, d in enumerate(devices):
            if d.get('max_output_channels', 0) > 0:
                lines.append(f"{idx:5d} | {d['name']} | {d['max_output_channels']}")
    else:
        lines.append("Index | HostAPI | Name | InCh | OutCh | Default SR")
        for idx, d in enumerate(devices):
            hai = int(d.get('hostapi', 0))
            ha = hostapis[hai]['name'] if 0 <= hai < len(hostapis) else str(hai)
            lines.append(
                f"{idx:5d} | {ha:7s} | {d['name'][:40]:40s} | {d.get('max_input_channels',0):4d} | {d.get('max_output_channels',0):5d} | {int(d.get('default_samplerate',0) or 0):7d}"
            )
    return "\n".join(lines)


def find_device_index_by_name(name_substring: str, require_output: bool = True, prefer_hostapi: Optional[str] = 'WASAPI') -> Optional[int]:
    name_l = name_substring.lower()
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()
    candidates = []
    for idx, d in enumerate(devices):
        if name_l in d['name'].lower():
            if require_output and d.get('max_output_channels', 0) <= 0:
                continue
            hai = int(d.get('hostapi', 0))
            ha = hostapis[hai]['name'] if 0 <= hai < len(hostapis) else ''
            score = 0
            if prefer_hostapi and prefer_hostapi.lower() in ha.lower():
                score += 10
            score += int(d.get('max_output_channels', 0))
            candidates.append((score, idx))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


# =============================
# Constants and utilities
# =============================

# Base subcarrier and rates
PILOT_HZ = 19000.0
STEREO_SUBCARRIER_HZ = 38000.0
RDS0_HZ = 57000.0
# Experimental RDS2 additional carriers (as per ETSI TS 103 634): 66.5/76/85.5 kHz
RDS2_SUBCARRIER_HZ = [66500.0, 76000.0, 85500.0]
RDS_BITRATE = 1187.5

# Levels (ratios). These are typical starting points; can be adjusted via CLI
DEFAULT_PILOT_LEVEL = 0.08
DEFAULT_RDS_LEVEL = 0.03
DEFAULT_RDS2_LEVEL = 0.01

# RDS2 experimental logo framing
RDS2_LOGO_MAX_W = 64
RDS2_LOGO_MAX_H = 32
RDS2_LOGO_MAGIC = 0xA7  # arbitrary marker for our simple framing

# RDS CRC generator polynomial g(x) = x^10 + x^8 + x^7 + x^5 + x^4 + x^3 + 1
# Polynomial value 0x5B9 (binary 101 1011 1001). Widely used in reference implementations.
RDS_CRC_POLY = 0x5B9
# Offset words for blocks A, B, C, D (10-bit) as commonly used
# Note: Some sources include C' with offset 0x1CC; we generate standard C for basic groups
RDS_OFFSET_A = 0x0FC
RDS_OFFSET_B = 0x198
RDS_OFFSET_C = 0x168
RDS_OFFSET_D = 0x1B4


def db_to_linear(db_value: float) -> float:
    return 10.0 ** (db_value / 20.0)


def clamp_audio(signal: np.ndarray, peak: float = 0.999) -> np.ndarray:
    return np.clip(signal, -peak, peak)


# =============================
# SciPy-free fallbacks
# =============================

def safe_resample_poly(y: np.ndarray, up: int, down: int) -> np.ndarray:
    if sp_resample_poly is not None:
        return sp_resample_poly(y, up=up, down=down)
    # Fallback: length-based linear interpolation
    new_len = int(round(y.shape[0] * up / float(down)))
    if new_len <= 1 or y.shape[0] <= 1:
        return y
    x = np.arange(y.shape[0], dtype=np.float64)
    xi = np.linspace(0.0, float(y.shape[0] - 1), new_len)
    if y.ndim == 1:
        yi = np.interp(xi, x, y.astype(np.float64))
        return yi.astype(y.dtype)
    else:
        # Resample per column
        out = np.zeros((new_len, y.shape[1]), dtype=np.float64)
        for ch in range(y.shape[1]):
            out[:, ch] = np.interp(xi, x, y[:, ch].astype(np.float64))
        return out.astype(y.dtype)


def safe_firwin(numtaps: int, cutoff: float, fs: float) -> np.ndarray:
    if sp_firwin is not None:
        return sp_firwin(numtaps, cutoff=cutoff, fs=fs)
    # Simple lowpass sinc with Hamming window
    n = np.arange(numtaps) - (numtaps - 1) / 2.0
    h = 2 * cutoff / fs * np.sinc(2 * cutoff / fs * n)
    w = np.hamming(numtaps)
    h *= w
    h /= np.sum(h)
    return h.astype(np.float64)


def safe_lfilter(b: np.ndarray, a: List[float], x: np.ndarray, axis: int = 0) -> np.ndarray:
    if sp_lfilter is not None:
        return sp_lfilter(b, a, x, axis=axis)
    # FIR-only fallback (a == [1.0]) via convolution, centered
    if not (len(a) == 1 and abs(float(a[0]) - 1.0) < 1e-12):
        raise RuntimeError("IIR filtering unsupported without SciPy")
    if x.ndim == 1:
        return np.convolve(x, b, mode='same')
    # Move axis to last, convolve per vector
    x_swap = np.moveaxis(x, axis, -1)
    y = np.empty_like(x_swap)
    for i in range(x_swap.shape[-1]):
        y[..., i] = np.convolve(x_swap[..., i], b, mode='same')
    return np.moveaxis(y, -1, axis)


# =============================
# RDS group building
# =============================

@dataclass
class RdsConfig:
    pi_code: int
    pty: int = 0
    tp: int = 0
    program_service_name: str = ""
    radiotext: str = ""


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


def _rds_block(word16: int, offset_word: int) -> Tuple[int, int]:
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

    # Block A: PI
    block_a = cfg.pi_code & 0xFFFF

    # Block B: Group type 0A: type=0, version A(0), TP, PTY, and segment address
    group_type = 0  # 0A
    version_a = 0
    tp = 1 if cfg.tp else 0
    pty = cfg.pty & 0x1F
    # Bits: 5-bit PTY, TP 1-bit, 4-bit group type+version, 5-bit segment/address
    # Formal layout (MSB..LSB): TP(1), PTY(5), Type(4), TA(1), MS(1), DI(1), Segment(2) etc for 0A
    # We'll follow common pack: 0AAAA BBBB includes various flags; keep TA=0, MS=0, DI=0
    # Here we use a simplified and commonly accepted packing for 0A:
    block_b = 0
    block_b |= (tp & 1) << 10
    block_b |= (pty & 0x1F) << 5
    block_b |= (group_type & 0xF) << 1
    block_b |= (version_a & 0x1)
    # Set address for PS segment in Block B least significant 2 bits
    block_b |= (segment_ch & 0x3)

    # Block C: 0A carries no additional info; set 0
    block_c = 0x0000

    # Block D: two characters of PS
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

    # A: PI
    block_a = cfg.pi_code & 0xFFFF

    # B: Group type=2, version A(0)
    group_type = 2
    version_a = 0
    tp = 1 if cfg.tp else 0
    pty = cfg.pty & 0x1F
    block_b = 0
    block_b |= (tp & 1) << 10
    block_b |= (pty & 0x1F) << 5
    block_b |= (group_type & 0xF) << 1
    block_b |= (version_a & 0x1)
    # Radiotext A/B flag = 0, text segment address 4 bits
    block_b |= (pair_idx & 0x0F)

    # C: two chars
    block_c = ((ord(c1) & 0xFF) << 8) | (ord(c2) & 0xFF)
    # D: two chars
    block_d = ((ord(c3) & 0xFF) << 8) | (ord(c4) & 0xFF)

    blocks = [
        _rds_block(block_a, RDS_OFFSET_A),
        _rds_block(block_b, RDS_OFFSET_B),
        _rds_block(block_c, RDS_OFFSET_C),
        _rds_block(block_d, RDS_OFFSET_D),
    ]
    return _pack_bits_from_blocks(blocks)


class RdsBitstreamGenerator:
    """Generate a continuous RDS bitstream (0/1) by cycling groups 0A and 2A."""

    def __init__(self, cfg: RdsConfig):
        self.cfg = cfg
        self.ps_index = 0
        self.rt_index = 0
        self.logo_frame: Optional[np.ndarray] = None
        self.logo_idx = 0

    def set_logo_bits(self, bits: Optional[np.ndarray]):
        self.logo_frame = bits
        self.logo_idx = 0

    def _next_logo_chunk(self, max_bits: int) -> Optional[np.ndarray]:
        if self.logo_frame is None or len(self.logo_frame) == 0:
            return None
        n = min(max_bits, len(self.logo_frame) - self.logo_idx)
        if n <= 0:
            # loop
            self.logo_idx = 0
            n = min(max_bits, len(self.logo_frame))
        chunk = self.logo_frame[self.logo_idx:self.logo_idx + n]
        self.logo_idx += n
        return chunk

    def next_group_bits(self) -> np.ndarray:
        # Insert small logo chunk periodically to keep presence in RDS2 sidebands
        if (self.ps_index % 5) == 0 and self.logo_frame is not None:
            # Return a slice of logo bits directly, not an RDS group
            chunk = self._next_logo_chunk(104)  # approx group size
            if chunk is not None:
                return chunk

        # Alternate two 0A per one 2A group to keep PS fresh
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
            out[filled:filled + n] = group[:n]
            filled += n
        return out


# =============================
# RDS BPSK waveform generation
# =============================

def differential_encode(bits: np.ndarray) -> np.ndarray:
    """Differential encoding for RDS: 1 -> phase invert, 0 -> no change. Output +/-1."""
    out = np.empty_like(bits, dtype=np.int8)
    phase = 1
    for i, b in enumerate(bits):
        if b:
            phase = -phase
        out[i] = phase
    return out.astype(np.float64)


def raised_cosine(num_taps: int, beta: float, sps: float) -> np.ndarray:
    """Raised cosine pulse shape filter (FIR) for BPSK shaping."""
    # Time vector centered
    t = (np.arange(num_taps) - (num_taps - 1) / 2.0) / sps
    h = np.zeros_like(t)
    for i, ti in enumerate(t):
        if abs(1 - (2 * beta * ti) ** 2) < 1e-8:
            # Special case to avoid division by zero
            h[i] = math.pi / 4 * np.sinc(1 / (2 * beta))
        else:
            h[i] = np.sinc(ti) * (np.cos(math.pi * beta * ti) / (1 - (2 * beta * ti) ** 2))
    # Normalize energy
    h = h / np.sum(h)
    return h


def bpsk_subcarrier(bits: np.ndarray, fs: float, subcarrier_hz: float, bitrate: float = RDS_BITRATE,
                    beta: float = 0.5, span_symbols: int = 6) -> np.ndarray:
    """Generate BPSK with differential encoding and raised-cosine shaping, mixed to subcarrier."""
    sps = fs / bitrate
    if sps < 4:
        raise ValueError("Sampling rate too low for RDS/RDS2")
    # Differential encoding to +/-1 symbols
    symbols = differential_encode(bits)
    # Upsample by inserting zeros between symbols
    up_factor = int(round(sps))
    if abs(sps - up_factor) > 1e-6:
        # Use fractional upsampling by polyphase filter later
        # Build discrete-time pulse at higher LCM rate using resample_poly
        base = np.repeat(symbols, int(math.ceil(sps)))
        sps_eff = int(math.ceil(sps))
    else:
        base = np.zeros(len(symbols) * up_factor)
        base[::up_factor] = symbols
        sps_eff = up_factor
    # Pulse shaping
    num_taps = max(41, int(span_symbols * sps_eff) | 1)
    h = raised_cosine(num_taps=num_taps, beta=beta, sps=float(sps_eff))
    shaped = np.convolve(base, h, mode='same')
    # If fractional, resample to exact fs
    if sps_eff != sps:
        shaped = safe_resample_poly(shaped, up=int(fs), down=int(bitrate * sps_eff))
        # The above is approximate and could be optimized; in practice, choose fs multiple of bitrate
    # Mix to subcarrier
    t = np.arange(len(shaped)) / fs
    carrier = np.cos(2 * np.pi * subcarrier_hz * t)
    return shaped * carrier


# =============================
# MPX generation
# =============================

def lowpass_stereo(audio: np.ndarray, fs: float, cutoff_hz: float = 15000.0) -> np.ndarray:
    numtaps = 513
    fir = safe_firwin(numtaps, cutoff=cutoff_hz, fs=fs)
    return safe_lfilter(fir, [1.0], audio, axis=0)


def make_mpx(
    left: np.ndarray,
    right: np.ndarray,
    fs: float,
    pilot_level: float = DEFAULT_PILOT_LEVEL,
    rds_level: float = DEFAULT_RDS_LEVEL,
    rds2_level: float = DEFAULT_RDS2_LEVEL,
    rds_bits: Optional[np.ndarray] = None,
    enable_rds2: bool = False,
) -> np.ndarray:
    assert left.shape == right.shape
    num_samples = left.shape[0]

    # Baseband L+R and L-R
    stereo = np.stack([left, right], axis=1)
    stereo = lowpass_stereo(stereo, fs)
    lpr = np.mean(stereo, axis=1)  # L+R
    lmr = stereo[:, 0] - stereo[:, 1]  # L-R

    t = np.arange(num_samples) / fs

    # 19 kHz pilot
    pilot = pilot_level * np.sin(2 * np.pi * PILOT_HZ * t)

    # 38 kHz DSB-SC for L-R
    stereo_sub = np.cos(2 * np.pi * STEREO_SUBCARRIER_HZ * t)
    dsb = lmr * stereo_sub

    # RDS at 57 kHz
    rds = np.zeros(num_samples)
    if rds_bits is not None and len(rds_bits) > 0:
        rds_wave = bpsk_subcarrier(rds_bits, fs=fs, subcarrier_hz=RDS0_HZ)
        rds[: len(rds_wave)] += rds_level * rds_wave[:num_samples]

    # RDS2 experimental
    if enable_rds2 and rds_bits is not None and len(rds_bits) > 0:
        for sc in RDS2_SUBCARRIER_HZ:
            rds2_wave = bpsk_subcarrier(rds_bits, fs=fs, subcarrier_hz=sc)
            rds[: len(rds2_wave)] += rds2_level * rds2_wave[:num_samples]

    mpx = lpr + pilot + dsb + rds
    return clamp_audio(mpx.astype(np.float32))


# =============================
# Audio I/O helpers
# =============================

def read_audio_file(path: str, target_fs: int) -> Tuple[np.ndarray, int]:
    data, fs = sf.read(path, always_2d=True)
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    if fs != target_fs:
        # Resample each channel
        left = safe_resample_poly(data[:, 0], up=target_fs, down=fs)
        right = safe_resample_poly(data[:, 1], up=target_fs, down=fs)
        data = np.stack([left, right], axis=1)
        fs = target_fs
    return data.astype(np.float32), fs


def generate_tone(duration_s: float, fs: int, freq_hz: float = 1000.0, level_db: float = -12.0) -> np.ndarray:
    t = np.arange(int(duration_s * fs)) / fs
    amp = db_to_linear(level_db)
    left = amp * np.sin(2 * np.pi * freq_hz * t)
    right = left.copy()
    return np.stack([left, right], axis=1).astype(np.float32)


class SystemLoopbackCapture:
    """Capture system audio via WASAPI loopback on Windows, or default input elsewhere.
    Auto-detects device channels and samplerate, and resamples to target fs.
    """

    def __init__(self, fs: int, device: Optional[int] = None, channels: int = 2, blocksize: int = 4096):
        self.fs = fs  # desired processing/output fs
        self.device = device
        self.req_channels = channels
        self.blocksize = blocksize
        self.q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=16)
        self._stream: Optional[sd.InputStream] = None
        self.captured_fs: Optional[float] = None
        self.captured_channels: int = channels
        self._hold: Optional[np.ndarray] = None  # buffer at captured_fs

    def start(self):
        extra = None
        if sys.platform.startswith('win'):
            try:
                extra = sd.WasapiSettings(loopback=True)
            except Exception:
                extra = None
        # Probe device info
        dev_info = None
        try:
            if self.device is not None:
                dev_info = sd.query_devices(self.device)
        except Exception:
            dev_info = None
        # Decide channels for loopback: use output channels if available
        ch = self.req_channels
        if dev_info is not None:
            out_ch = int(dev_info.get('max_output_channels', 0) or 0)
            if out_ch > 0:
                ch = min(max(1, out_ch), 2)
        # Decide samplerate: try desired fs first, else device default
        sr_try = [self.fs]
        if dev_info is not None:
            try:
                sr_def = float(dev_info.get('default_samplerate') or 0)
                if sr_def and int(sr_def) != self.fs:
                    sr_try.append(int(sr_def))
            except Exception:
                pass
        opened = False
        err_last = None

        def cb(indata, frames, time_info, status):
            try:
                x = indata.copy()
                if x.ndim == 1:
                    x = np.stack([x, x], axis=1)
                elif x.shape[1] == 1:
                    x = np.repeat(x, 2, axis=1)
                self.q.put_nowait(x.astype(np.float32))
            except queue.Full:
                pass

        channels_try = []
        if dev_info is not None:
            out_ch = int(dev_info.get('max_output_channels', 0) or 0)
            if out_ch:
                channels_try.extend([min(out_ch, 2), out_ch])
        channels_try.extend([2, 1])
        tried = set()
        for sr in sr_try:
            for ch_try in channels_try:
                key = (sr, ch_try)
                if key in tried:
                    continue
                tried.add(key)
                try:
                    self._stream = sd.InputStream(
                        samplerate=sr,
                        device=self.device,
                        channels=ch_try,
                        dtype='float32',
                        callback=cb,
                        blocksize=self.blocksize,
                        extra_settings=extra,
                    )
                    self._stream.start()
                    self.captured_fs = float(sr)
                    self.captured_channels = ch_try
                    opened = True
                    break
                except Exception as e:
                    err_last = e
                    self._stream = None
            if opened:
                break
        if not opened:
            raise RuntimeError(f"Failed to open loopback input: {err_last}")

    def read(self, frames: int) -> np.ndarray:
        """Return 'frames' at target fs, stereo float32."""
        if self.captured_fs is None:
            return np.zeros((frames, 2), dtype=np.float32)
        # Accumulate enough source samples
        need_src = int(round(frames * self.captured_fs / float(self.fs)))
        if need_src <= 0:
            need_src = 1
        if self._hold is None:
            self._hold = np.zeros((0, 2), dtype=np.float32)
        while self._hold.shape[0] < need_src:
            try:
                buf = self.q.get(timeout=1.0)
            except queue.Empty:
                buf = np.zeros((min(self.blocksize, need_src), 2), dtype=np.float32)
            # Ensure stereo
            if buf.ndim == 1:
                buf = np.stack([buf, buf], axis=1)
            elif buf.shape[1] == 1:
                buf = np.repeat(buf, 2, axis=1)
            self._hold = np.vstack([self._hold, buf])
        src = self._hold[:need_src, :]
        self._hold = self._hold[need_src:, :]
        if int(self.captured_fs) == int(self.fs):
            # Same rate: just match frame count
            if src.shape[0] < frames:
                pad = np.zeros((frames - src.shape[0], 2), dtype=np.float32)
                src = np.vstack([src, pad])
            elif src.shape[0] > frames:
                src = src[:frames, :]
            return src.astype(np.float32)
        # Resample to exact frames
        left = safe_resample_poly(src[:, 0], up=self.fs, down=int(self.captured_fs))
        right = safe_resample_poly(src[:, 1], up=self.fs, down=int(self.captured_fs))
        y = np.stack([left, right], axis=1)
        # Adjust length
        if y.shape[0] < frames:
            pad = np.zeros((frames - y.shape[0], 2), dtype=np.float32)
            y = np.vstack([y, pad])
        elif y.shape[0] > frames:
            y = y[:frames, :]
        return y.astype(np.float32)

    def stop(self):
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None
        with self.q.mutex:
            self.q.queue.clear()
        self._hold = None


# =============================
# CLI
# =============================

@click.group()
def cli():
    """FM MPX generator with RDS/RDS2. Outputs to soundcard or file."""


@cli.command()
@click.option("--fs", type=int, default=192000, show_default=True, help="Sample rate for MPX output")
@click.option("--verbose", is_flag=True, default=False, help="Show host API, in/out channels, default SR")
def devices(fs: int, verbose: bool):
    """List audio output devices (add --verbose for host API details)."""
    sd.default.samplerate = fs
    click.echo(list_devices(verbose=verbose))


def _prepare_rds_bits(pi: int, ps: str, rt: str, seconds: float, fs: int) -> np.ndarray:
    cfg = RdsConfig(pi_code=pi, program_service_name=ps or "", radiotext=rt or "")
    gen = RdsBitstreamGenerator(cfg)
    total_bits = int(math.ceil(seconds * RDS_BITRATE * 1.1))  # some headroom
    return gen.generate_bits(total_bits)


@cli.command()
@click.option("--input", type=click.Path(exists=True, dir_okay=False), help="Stereo WAV/FLAC/AIFF input file")
@click.option("--tone", type=float, default=None, help="If set, generate a sine tone at this frequency (Hz)")
@click.option("--duration", type=float, default=30.0, show_default=True, help="Duration if using tone (s)")
@click.option("--fs", type=int, default=192000, show_default=True, help="Sample rate for MPX output")
@click.option("--device", type=int, default=None, help="Sounddevice output index")
@click.option("--pi", type=str, default="0x1234", show_default=True, help="PI code, hex like 0x1234")
@click.option("--ps", type=str, default="TESTFM", show_default=True, help="Program Service name (8 chars)")
@click.option("--rt", type=str, default="", help="Radiotext (up to 64 chars)")
@click.option("--pilot-level", type=float, default=DEFAULT_PILOT_LEVEL, show_default=True, help="Pilot level (linear)")
@click.option("--rds-level", type=float, default=DEFAULT_RDS_LEVEL, show_default=True, help="RDS level (linear)")
@click.option("--rds2", is_flag=True, default=False, help="Enable experimental RDS2 sidebands")
@click.option("--rds2-level", type=float, default=DEFAULT_RDS2_LEVEL, show_default=True, help="RDS2 per-subcarrier level (linear)")
@click.option("--logo", type=click.Path(exists=True, dir_okay=False), default=None, help="Path to station logo image (png/jpg)")
@click.option("--level-mpx", type=float, default=0.0, show_default=True, help="Overall MPX gain (dB)")
@click.option("--blocksize", type=int, default=4096, show_default=True, help="Block size for streaming frames")
@click.option("--system-audio", is_flag=True, default=False, help="Capture system audio (WASAPI loopback on Windows)")
@click.option("--capture-device", type=int, default=None, help="Capture device index for system audio (WASAPI output device)")
@click.option("--capture-name", type=str, default=None, help="Capture device name substring (prefers WASAPI)")
@click.option("--device-name", type=str, default=None, help="Output device name substring (prefers WASAPI)")
def play(input: Optional[str], tone: Optional[float], duration: float, fs: int, device: Optional[int], pi: str, ps: str,
         rt: str, pilot_level: float, rds_level: float, rds2: bool, rds2_level: float, logo: Optional[str], level_mpx: float, blocksize: int,
         system_audio: bool, capture_device: Optional[int], capture_name: Optional[str], device_name: Optional[str]):
    """Play composite MPX with RDS/RDS2 to a sound device."""
    if not system_audio and input is None and tone is None:
        raise click.UsageError("Provide --input or --tone or --system-audio")

    # Resolve by name if provided
    if capture_name and capture_device is None:
        idx = find_device_index_by_name(capture_name, require_output=True, prefer_hostapi='WASAPI')
        if idx is None:
            raise click.BadParameter(f"Capture device with name containing '{capture_name}' not found")
        capture_device = idx
    if device_name and device is None:
        idx = find_device_index_by_name(device_name, require_output=True, prefer_hostapi='WASAPI')
        if idx is None:
            raise click.BadParameter(f"Output device with name containing '{device_name}' not found")
        device = idx

    sd.default.samplerate = fs
    if device is not None:
        sd.default.device = device

    cfg = RdsConfig(pi_code=int(pi, 16), program_service_name=ps or "", radiotext=rt or "")
    gen = RdsBitstreamGenerator(cfg)
    if rds2 and logo:
        gen.set_logo_bits(load_logo_bits(logo))

    # Prepare audio source
    stereo: Optional[np.ndarray] = None
    capture: Optional[SystemLoopbackCapture] = None
    if system_audio:
        capture = SystemLoopbackCapture(fs=fs, device=capture_device, channels=2, blocksize=blocksize)
        capture.start()
    elif input:
        stereo, _ = read_audio_file(input, target_fs=fs)
    else:
        stereo = generate_tone(duration_s=duration, fs=fs, freq_hz=tone or 1000.0)

    # Initial RDS bits
    total_seconds = 2.0 if system_audio else (stereo.shape[0] / fs)
    rds_bits = gen.generate_bits(int(math.ceil(total_seconds * RDS_BITRATE * 1.1)))

    q_out: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=8)

    def producer_from_array():
        assert stereo is not None
        idx = 0
        gain = db_to_linear(level_mpx)
        while idx < stereo.shape[0]:
            end = min(idx + blocksize, stereo.shape[0])
            left = stereo[idx:end, 0]
            right = stereo[idx:end, 1]
            bits_needed = int(math.ceil((end - idx) / fs * RDS_BITRATE)) + 208
            if len(rds_bits) < bits_needed:
                extra = gen.generate_bits(int(math.ceil(2.0 * RDS_BITRATE)))
                rds_bits[:] = np.concatenate([rds_bits, extra])
            bits_block = rds_bits[:bits_needed]
            rds_bits[:] = rds_bits[bits_needed:]

            mpx = make_mpx(left, right, fs, pilot_level, rds_level, rds2_level, bits_block, rds)
            mpx *= gain
            q_out.put(mpx, block=True)
            idx = end
        q_out.put(None)

    def producer_from_capture():
        assert capture is not None
        gain = db_to_linear(level_mpx)
        while True:
            buf = capture.read(blocksize)
            left = buf[:, 0]
            right = buf[:, 1]
            bits_needed = int(math.ceil((len(left)) / fs * RDS_BITRATE)) + 208
            if len(rds_bits) < bits_needed:
                extra = gen.generate_bits(int(math.ceil(2.0 * RDS_BITRATE)))
                rds_bits[:] = np.concatenate([rds_bits, extra])
            bits_block = rds_bits[:bits_needed]
            rds_bits[:] = rds_bits[bits_needed:]

            mpx = make_mpx(left, right, fs, pilot_level, rds_level, rds2_level, bits_block, rds)
            mpx *= gain
            try:
                q_out.put(mpx, timeout=1.0)
            except queue.Full:
                pass

    # Choose producer
    import threading
    if system_audio:
        prod_thread = threading.Thread(target=producer_from_capture, daemon=True)
    else:
        prod_thread = threading.Thread(target=producer_from_array, daemon=True)
    prod_thread.start()

    def callback(outdata, frames, time_info, status):
        try:
            chunk = q_out.get(timeout=1.0)
        except queue.Empty:
            outdata[:] = 0
            return
        if chunk is None:
            raise sd.CallbackStop
        if outdata.shape[1] == 1:
            outdata[:, 0] = chunk[:frames] if len(chunk) >= frames else np.pad(chunk, (0, frames - len(chunk)))
        else:
            mono = chunk[:frames] if len(chunk) >= frames else np.pad(chunk, (0, frames - len(chunk)))
            outdata[:, 0] = mono
            outdata[:, 1] = mono

    try:
        with sd.OutputStream(channels=1, dtype='float32', callback=callback, blocksize=blocksize):
            while prod_thread.is_alive():
                time.sleep(0.1)
    finally:
        if capture is not None:
            capture.stop()


@cli.command()
@click.option("--output", type=click.Path(dir_okay=False), required=True, help="Output WAV path for MPX")
@click.option("--input", type=click.Path(exists=True, dir_okay=False), help="Stereo WAV/FLAC/AIFF input file")
@click.option("--tone", type=float, default=None, help="If set, generate a sine tone at this frequency (Hz)")
@click.option("--duration", type=float, default=30.0, show_default=True, help="Duration if using tone (s)")
@click.option("--fs", type=int, default=192000, show_default=True, help="Sample rate for MPX output")
@click.option("--pi", type=str, default="0x1234", show_default=True, help="PI code, hex like 0x1234")
@click.option("--ps", type=str, default="TESTFM", show_default=True, help="Program Service name (8 chars)")
@click.option("--rt", type=str, default="", help="Radiotext (up to 64 chars)")
@click.option("--pilot-level", type=float, default=DEFAULT_PILOT_LEVEL, show_default=True, help="Pilot level (linear)")
@click.option("--rds-level", type=float, default=DEFAULT_RDS_LEVEL, show_default=True, help="RDS level (linear)")
@click.option("--rds2", is_flag=True, default=False, help="Enable experimental RDS2 sidebands")
@click.option("--rds2-level", type=float, default=DEFAULT_RDS2_LEVEL, show_default=True, help="RDS2 per-subcarrier level (linear)")
@click.option("--logo", type=click.Path(exists=True, dir_okay=False), default=None, help="Path to station logo image (png/jpg)")
@click.option("--level-mpx", type=float, default=0.0, show_default=True, help="Overall MPX gain (dB)")
def tofile(output: str, input: Optional[str], tone: Optional[float], duration: float, fs: int, pi: str, ps: str, rt: str,
           pilot_level: float, rds_level: float, rds2: bool, rds2_level: float, logo: Optional[str], level_mpx: float):
    """Render composite MPX with RDS/RDS2 to a WAV file (mono)."""
    if input is None and tone is None:
        raise click.UsageError("Provide --input or --tone")

    if input:
        stereo, _ = read_audio_file(input, target_fs=fs)
    else:
        stereo = generate_tone(duration_s=duration, fs=fs, freq_hz=tone or 1000.0)

    total_seconds = stereo.shape[0] / fs

    cfg = RdsConfig(pi_code=int(pi, 16), program_service_name=ps or "", radiotext=rt or "")
    gen = RdsBitstreamGenerator(cfg)

    if rds2 and logo:
        gen.set_logo_bits(load_logo_bits(logo))

    rds_bits = gen.generate_bits(int(math.ceil(total_seconds * RDS_BITRATE * 1.1)))

    mpx = make_mpx(
        left=stereo[:, 0],
        right=stereo[:, 1],
        fs=fs,
        pilot_level=pilot_level,
        rds_level=rds_level,
        rds2_level=rds2_level,
        rds_bits=rds_bits,
        enable_rds2=rds2,
    )
    mpx *= db_to_linear(level_mpx)

    sf.write(output, mpx, samplerate=fs, subtype='PCM_24')
    click.echo(f"Wrote {output} ({len(mpx)/fs:.2f}s at {fs} Hz)")


def load_logo_bits(path: str) -> np.ndarray:
    """Load an image and pack as a simple framed monochrome bitstream for RDS2.
    Handles transparency by compositing onto white, then converting to L.
    """
    img = Image.open(path)
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        bg.alpha_composite(img)
        img = bg.convert("L")
    else:
        img = img.convert('L')
    w = min(RDS2_LOGO_MAX_W, max(1, img.width))
    h = min(RDS2_LOGO_MAX_H, max(1, img.height))
    if img.width != w or img.height != h:
        img = img.resize((w, h), Image.LANCZOS)
    arr = np.array(img)
    thr = float(arr.mean())
    bits = (arr >= thr).astype(np.uint8)

    header = []
    def put(val: int, nbits: int):
        for i in range(nbits - 1, -1, -1):
            header.append((val >> i) & 1)

    put(RDS2_LOGO_MAGIC, 8)
    put(w, 7)
    put(h, 6)
    put(0, 3)

    payload_bits = bits.flatten().tolist()

    payload_bytes = []
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
    def put_footer(val: int, nbits: int):
        for i in range(nbits - 1, -1, -1):
            header.append((val >> i) & 1)
    put_footer(checksum, 16)

    all_bits = np.array(header + payload_bits, dtype=np.uint8)
    return all_bits


if __name__ == "__main__":
    cli()