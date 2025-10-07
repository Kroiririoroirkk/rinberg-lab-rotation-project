from pathlib import Path
from tkinter.font import Font
from typing import Final

import matplotlib

# File paths
ROI_ZIP_PATH: Final[Path] = Path(
    'data/sebolddata/GLOM/ROIs/RoiSet_ET_2025-10-02.zip')
TIF_FOLDER: Final[Path] = Path('data/sebolddata/GLOM/TIFFs')
H5_PATH: Final[Path] = Path(
    'data/sebolddata/GLOM/mouse0953_sess20_D2025_8_18T12_36_43.h5')
H5_PICKLE_PATH: Final[Path] = Path('metadata.pkl')
TIF_EXPORT_PATH: Final[Path] = Path('export.tif')

# Calcium imaging playback settings
NUM_FRAMES_BUYIN: Final[int] = 5
NUM_FRAMES_AVG: Final[int] = 30
BASELINE_AVG_START: Final[int] = 0
BASELINE_AVG_END: Final[int] = 20
SLOW_SPEED: Final[float] = 0.2
DEFAULT_SPATIAL_BLUR: Final[float] = 1.0
DEFAULT_TEMPORAL_BLUR: Final[int] = 5
MIN_DELTA_F: Final[float] = -0.5
MAX_DELTA_F: Final[float] = 2.0
HEAT_CMAP: Final[matplotlib.colors.Colormap] = matplotlib.colormaps['inferno']

# Tkinter format settings
DEFAULT_WINDOW_WIDTH: Final[int] = 1600
DEFAULT_WINDOW_HEIGHT: Final[int] = 850
DEFAULT_WINDOW_SIZE: Final[
    str] = f'{DEFAULT_WINDOW_WIDTH}x{DEFAULT_WINDOW_HEIGHT}'
SMALL_PAD: Final[int] = 10
MEDIUM_PAD: Final[int] = 20
LARGE_PAD: Final[int] = 30
HEADING_FONT: Final[Font] = ('', 18)

# Miscellaneous
JOB_SCHEDULE_DELAY: Final[int] = 1000
