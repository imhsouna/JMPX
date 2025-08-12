# Python FM MPX + RDS/RDS2 Streaming CLI

Cross-platform (Windows/Linux) Python CLI that generates an FM composite (MPX) signal with stereo pilot, stereo DSB-SC subcarrier, RDS (57 kHz) and optional RDS2 subcarriers, and streams to a soundcard or writes a WAV file. Inspired by JMPX.

Note: Transmitting RF may be illegal without a license. This tool outputs baseband MPX audio. Use responsibly.

## Quickstart

- Python 3.10+
- Soundcard supporting 192 kHz (recommended) for proper RDS spectral placement

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m rds2 --help
```

Legacy entrypoint is still available:

```bash
python rds2_stream.py --help
```

## Examples

- Play a stereo WAV and inject basic RDS PS/PI at 192 kHz to default output:
```bash
python -m rds2 play --input my_stereo.wav --fs 192000 --pi 0x1234 --ps "TESTFM" --rt "Hello world" --level-mpx -3
```

- Generate a 1 kHz tone and write MPX WAV with RDS:
```bash
python -m rds2 tofile --output mpx.wav --duration 30 --tone 1000 --fs 192000 --pi 0x1234 --ps "TEST" --rt "Demo" --rds2
```

- List audio devices and pick one:
```bash
python -m rds2 devices
python -m rds2 play --device 3 --input my.wav --fs 192000 --pi 0x1234 --ps "STATION"
```

## Features

- FM MPX generation: L+R baseband, 19 kHz pilot, L-R DSB-SC at 38 kHz
- RDS (RBDS) at 57 kHz BPSK with CRC and differential encoding
- Optional RDS2 sidebands (SCA at 66.5/76/85.5 kHz) experimental
- Output to soundcard (real-time) or WAV file
- CLI with Click

## Limitations

- RDS2 implementation is experimental and may not be fully standard-compliant
- Requires high sample rate (>= 192 kHz) for clean spectral placement
- This produces MPX audio. RF transmission requires external hardware and legal authorization

## License

MIT

