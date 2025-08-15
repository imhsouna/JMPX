#!/usr/bin/env python3
import warnings

from rds2.cli import cli

if __name__ == "__main__":
    warnings.warn(
        "rds2_stream.py is deprecated. Use `python -m rds2` or `rds2` entrypoint instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    cli()