"""
Global Configuration for ANC Simulation Project
"""

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
SRC_DIR = PROJECT_ROOT / 'src'
OUTPUT_DIR = PROJECT_ROOT / 'output'

# Audio parameters
DEFAULT_SAMPLE_RATE = 16000  # Hz
DEFAULT_BIT_DEPTH = 16

# Frequency range of interest (car interior noise)
MIN_FREQUENCY = 20   # Hz
MAX_FREQUENCY = 300  # Hz

# FxLMS default parameters
FXLMS_FILTER_LENGTH = 256
FXLMS_STEP_SIZE = 0.001
FXLMS_LEAKAGE = 0.0

# Acoustic model defaults
PRIMARY_PATH_TAPS = 256
SECONDARY_PATH_TAPS = 128
SECONDARY_PATH_ERROR = 0.05  # 5% modeling error

# Simulation defaults
DEFAULT_DURATION = 10.0  # seconds

# Visualization defaults
FIGURE_DPI = 150
ANIMATION_FPS = 30
SPECTROGRAM_NPERSEG = 512
