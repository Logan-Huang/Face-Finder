"""Allows `python -m facefinder ...`."""

import sys

from .cli import main

sys.exit(main())
