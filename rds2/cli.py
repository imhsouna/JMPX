from __future__ import annotations

import logging
import math
import queue
import threading
import time
from typing import Optional

import click
import numpy as np
try:
    import sounddevice as sd
except Exception:  # PortAudio may be missing in some environments
    sd = None  # type: ignore

from .config import (
    DEFAULT_PILOT_LEVEL,
    DEFAULT_RDS_LEVEL,
    DEFAULT_RDS2_LEVEL,
    RDS_BITRATE,
)
from .rds import RdsBitstreamGenerator
from .audio_io import read_audio_file, generate_tone, find_device_index_by_name
from .mpx import make_mpx
from .logo import load_logo_bits
from .config import db_to_linear


def _setup_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


@click.group()
@click.option("--verbose", "verbosity", count=True, help="Increase verbosity (-v, -vv)")
def cli(verbosity: int) -> None:
    """FM MPX generator with RDS/RDS2. Outputs to soundcard or file."""
    _setup_logging(verbosity)


@cli.command()
@click.option("--fs", type=int, default=192000, show_default=True, help="Sample rate for MPX output")
def devices(fs: int) -> None:
    """List audio output devices."""
    if sd is None:
        raise click.UsageError("sounddevice/PortAudio not available in this environment")
    sd.default.samplerate = fs
    info = sd.query_devices()
    click.echo("Index | Name | Max output channels | Max input channels")
    for idx, dev in enumerate(info):
        click.echo(f"{idx:5d} | {dev['name']} | {dev.get('max_output_channels', 0)} | {dev.get('max_input_channels', 0)}")


@cli.command()
@click.option("--input", type=click.Path(exists=True, dir_okay=False), help="Stereo WAV/FLAC/AIFF input file")
@click.option("--tone", type=float, default=None, help="If set, generate a sine tone at this frequency (Hz)")
@click.option("--duration", type=float, default=30.0, show_default=True, help="Duration if using tone (s)")
@click.option("--fs", type=int, default=192000, show_default=True, help="Sample rate for MPX output")
@click.option("--device", type=int, default=None, help="Sounddevice output index")
@click.option("--device-name", type=str, default=None, help="Output device name (substring match)")
@click.option("--system-audio", is_flag=True, default=False, help="Capture system audio via WASAPI loopback (Windows)")
@click.option("--capture-name", type=str, default=None, help="Playback or input device name to capture from")
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
def play(
    input: Optional[str],
    tone: Optional[float],
    duration: float,
    fs: int,
    device: Optional[int],
    device_name: Optional[str],
    system_audio: bool,
    capture_name: Optional[str],
    pi: str,
    ps: str,
    rt: str,
    pilot_level: float,
    rds_level: float,
    rds2: bool,
    rds2_level: float,
    logo: Optional[str],
    level_mpx: float,
    blocksize: int,
) -> None:
    """Play composite MPX with RDS/RDS2 to a sound device."""
    if input is None and tone is None and not (system_audio or capture_name):
        raise click.UsageError("Provide --input or --tone, or use --system-audio/--capture-name for live capture")

    if sd is None:
        raise click.UsageError("sounddevice/PortAudio not available in this environment")
    sd.default.samplerate = fs

    if device_name and device is None:
        idx = find_device_index_by_name(device_name, is_output=True)
        if idx is None:
            raise click.UsageError(f"Output device not found by name: {device_name}")
        device = idx
    if device is not None:
        sd.default.device = (
            sd.default.device[0] if isinstance(sd.default.device, (list, tuple)) else None,
            device,
        )

    cfg = RdsBitstreamGenerator(
        cfg=__import__("rds2.config", fromlist=["RdsConfig"]).RdsConfig(
            pi_code=int(pi, 16), program_service_name=ps or "", radiotext=rt or ""
        )
    )
    if rds2 and logo:
        cfg.set_logo_bits(load_logo_bits(logo))

    gain = db_to_linear(level_mpx)

    if system_audio or capture_name:
        cap_idx: Optional[int] = None
        extra_settings = None

        if system_audio:
            try:
                extra_settings = sd.WasapiSettings(loopback=True)
            except Exception as exc:  # pragma: no cover
                raise click.UsageError("--system-audio requires Windows WASAPI (sounddevice.WasapiSettings)") from exc
            if capture_name:
                cap_idx = find_device_index_by_name(capture_name, is_output=True)
                if cap_idx is None:
                    raise click.UsageError(f"Playback device for loopback not found: {capture_name}")
        else:
            if capture_name:
                cap_idx = find_device_index_by_name(capture_name, is_output=False)
                if cap_idx is None:
                    raise click.UsageError(f"Input device not found: {capture_name}")

        q_in: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=16)
        q_out: "queue.Queue[Optional[np.ndarray]]" = queue.Queue(maxsize=16)

        def in_callback(indata, frames, time_info, status):
            try:
                data = indata if indata.shape[1] == 2 else np.repeat(indata, 2, axis=1)
                q_in.put_nowait(data.copy())
            except queue.Full:  # pragma: no cover
                pass

        def worker_stop_condition(start_time: float) -> bool:
            if tone is not None or input is not None:
                return True
            if duration and duration > 0:
                return (time.time() - start_time) >= duration
            return False

        def worker():
            start_time = time.time()
            while True:
                if worker_stop_condition(start_time):
                    q_out.put(None)
                    return
                try:
                    stereo_block = q_in.get(timeout=0.5)
                except queue.Empty:
                    continue
                frames_local = stereo_block.shape[0]
                bits_needed = int(math.ceil(frames_local / fs * RDS_BITRATE)) + 208
                bits_block = cfg.generate_bits(bits_needed)
                mpx = make_mpx(
                    left=stereo_block[:, 0],
                    right=stereo_block[:, 1],
                    fs=fs,
                    pilot_level=pilot_level,
                    rds_level=rds_level,
                    rds2_level=rds2_level,
                    rds_bits=bits_block,
                    enable_rds2=rds2,
                )
                mpx *= gain
                q_out.put(mpx, block=True)

        def out_callback(outdata, frames, time_info, status):
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

        worker_thread = threading.Thread(target=worker, daemon=True)
        worker_thread.start()

        with sd.InputStream(
            device=cap_idx,
            channels=2,
            dtype='float32',
            callback=in_callback,
            blocksize=blocksize,
            samplerate=fs,
            extra_settings=extra_settings,
        ), sd.OutputStream(
            channels=1,
            dtype='float32',
            callback=out_callback,
            blocksize=blocksize,
            samplerate=fs,
        ):
            try:
                while worker_thread.is_alive():
                    time.sleep(0.1)
            except KeyboardInterrupt:  # pragma: no cover
                pass
        return

    if input:
        stereo, _ = read_audio_file(input, target_fs=fs)
    else:
        stereo = generate_tone(duration_s=duration, fs=fs, freq_hz=tone or 1000.0)

    total_seconds = stereo.shape[0] / fs

    rds_bits = cfg.generate_bits(int(math.ceil(total_seconds * RDS_BITRATE * 1.1)))

    q_out: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=8)

    def producer():
        idx = 0
        nonlocal rds_bits
        while idx < stereo.shape[0]:
            end = min(idx + blocksize, stereo.shape[0])
            left = stereo[idx:end, 0]
            right = stereo[idx:end, 1]
            bits_needed = int(math.ceil((end - idx) / fs * RDS_BITRATE)) + 208
            if len(rds_bits) < bits_needed:
                extra = cfg.generate_bits(int(math.ceil(2.0 * RDS_BITRATE)))
                rds_bits = np.concatenate([rds_bits, extra])
            bits_block = rds_bits[:bits_needed]
            rds_bits = rds_bits[bits_needed:]

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
        q_out.put(None)

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

    prod_thread = threading.Thread(target=producer, daemon=True)
    prod_thread.start()

    with sd.OutputStream(channels=1, dtype='float32', callback=callback, blocksize=blocksize, samplerate=fs):
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
def tofile(
    output: str,
    input: Optional[str],
    tone: Optional[float],
    duration: float,
    fs: int,
    pi: str,
    ps: str,
    rt: str,
    pilot_level: float,
    rds_level: float,
    rds2: bool,
    rds2_level: float,
    logo: Optional[str],
    level_mpx: float,
) -> None:
    """Render composite MPX with RDS/RDS2 to a WAV file (mono)."""
    if input is None and tone is None:
        raise click.UsageError("Provide --input or --tone")

    if input:
        stereo, _ = read_audio_file(input, target_fs=fs)
    else:
        stereo = generate_tone(duration_s=duration, fs=fs, freq_hz=tone or 1000.0)

    total_seconds = stereo.shape[0] / fs

    cfg = RdsBitstreamGenerator(
        cfg=__import__("rds2.config", fromlist=["RdsConfig"]).RdsConfig(
            pi_code=int(pi, 16), program_service_name=ps or "", radiotext=rt or ""
        )
    )

    if rds2 and logo:
        cfg.set_logo_bits(load_logo_bits(logo))

    rds_bits = cfg.generate_bits(int(math.ceil(total_seconds * RDS_BITRATE * 1.1)))

    from soundfile import write as sf_write

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

    sf_write(output, mpx, samplerate=fs, subtype='PCM_24')
    click.echo(f"Wrote {output} ({len(mpx)/fs:.2f}s at {fs} Hz)")