"""Data and output paths, independent of the invoking working directory."""
from __future__ import annotations

import os
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = Path(os.environ.get("DYNAMICS_PREDICTION_ROOT", SOURCE_ROOT)).expanduser().resolve()
