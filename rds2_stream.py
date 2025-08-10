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
from scipy.signal import resample_poly, firwin, lfilter
from PIL import Image


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
        shaped = resample_poly(shaped, up=int(fs), down=int(bitrate * sps_eff))
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
    fir = firwin(numtaps, cutoff=cutoff_hz, fs=fs)
    return lfilter(fir, [1.0], audio, axis=0)


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


# =============================
# CLI
# =============================

@click.group()
def cli():
    """FM MPX generator with RDS/RDS2. Outputs to soundcard or file."""


@cli.command()
@click.option("--fs", type=int, default=192000, show_default=True, help="Sample rate for MPX output")
def devices(fs: int):
    """List audio output devices."""
    sd.default.samplerate = fs
    info = sd.query_devices()
    click.echo("Index | Name | Max output channels")
    for idx, dev in enumerate(info):
        if dev.get("max_output_channels", 0) > 0:
            click.echo(f"{idx:5d} | {dev['name']} | {dev['max_output_channels']}")


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
def play(input: Optional[str], tone: Optional[float], duration: float, fs: int, device: Optional[int], pi: str, ps: str,
         rt: str, pilot_level: float, rds_level: float, rds2: bool, rds2_level: float, logo: Optional[str], level_mpx: float, blocksize: int):
    """Play composite MPX with RDS/RDS2 to a sound device."""
    if input is None and tone is None:
        raise click.UsageError("Provide --input or --tone")

    sd.default.samplerate = fs
    if device is not None:
        sd.default.device = device

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

    # Streaming in blocks
    q_out: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=8)

    def producer():
        idx = 0
        gain = db_to_linear(level_mpx)
        while idx < stereo.shape[0]:
            end = min(idx + blocksize, stereo.shape[0])
            left = stereo[idx:end, 0]
            right = stereo[idx:end, 1]
            # bits for this block (approx)
            bits_needed = int(math.ceil((end - idx) / fs * RDS_BITRATE)) + 208
            if len(rds_bits) < bits_needed:
                # extend
                extra = gen.generate_bits(int(math.ceil(2.0 * RDS_BITRATE)))
                rds_bits[:] = np.concatenate([rds_bits, extra])
            bits_block = rds_bits[:bits_needed]
            rds_bits[:] = rds_bits[bits_needed:]

            mpx = make_mpx(
                left=left,
                right=right,
                fs=fs,
                pilot_level=pilot_level,
                rds_level=rds_level,
                rds2_level=rds2_level,
                rds_bits=bits_block,
                enable_rds2=rds2,
            )
            mpx *= gain
            q_out.put(mpx, block=True)
            idx = end
        # signal end
        q_out.put(None)

    def callback(outdata, frames, time_info, status):
        try:
            chunk = q_out.get(timeout=1.0)
        except queue.Empty:
            outdata[:] = 0
            return
        if chunk is None:
            raise sd.CallbackStop
        # outdata is 2D (frames, channels). We output mono MPX to 1 ch; if device requires stereo, duplicate.
        if outdata.shape[1] == 1:
            outdata[:, 0] = chunk[:frames] if len(chunk) >= frames else np.pad(chunk, (0, frames - len(chunk)))
        else:
            mono = chunk[:frames] if len(chunk) >= frames else np.pad(chunk, (0, frames - len(chunk)))
            outdata[:, 0] = mono
            outdata[:, 1] = mono

    # Run
    import threading
    prod_thread = threading.Thread(target=producer, daemon=True)
    prod_thread.start()

    with sd.OutputStream(channels=1, dtype='float32', callback=callback, blocksize=blocksize):
        while prod_thread.is_alive():
            time.sleep(0.1)


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
    Frame format (repeating):
    - 8 bits magic (0xA7)
    - 7 bits width (1..64)
    - 6 bits height (1..32)
    - 3 bits reserved (0)
    - width*height bits, row-major, 1=white, 0=black
    - 16 bits simple checksum (sum of payload bytes & 0xFFFF)
    This is not an ETSI RDS2 logo standard; it's a practical, receiver-agnostic payload carried on RDS2 BPSK.
    """
    img = Image.open(path).convert('L')
    w = min(RDS2_LOGO_MAX_W, max(1, img.width))
    h = min(RDS2_LOGO_MAX_H, max(1, img.height))
    if img.width != w or img.height != h:
        img = img.resize((w, h), Image.LANCZOS)
    arr = np.array(img)
    # Binarize with Otsu-like threshold (simple mean)
    thr = float(arr.mean())
    bits = (arr >= thr).astype(np.uint8)

    # Header bits
    header = []
    def put(val: int, nbits: int):
        for i in range(nbits - 1, -1, -1):
            header.append((val >> i) & 1)

    put(RDS2_LOGO_MAGIC, 8)
    put(w, 7)
    put(h, 6)
    put(0, 3)

    payload_bits = bits.flatten().tolist()

    # Compute checksum over payload bytes
    # Pack payload bits into bytes MSB-first
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
    footer = []
    put(checksum, 16)

    all_bits = np.array(header + payload_bits + footer, dtype=np.uint8)
    return all_bits


if __name__ == "__main__":
    cli()