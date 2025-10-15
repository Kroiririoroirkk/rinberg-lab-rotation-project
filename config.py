from collections import OrderedDict
from pathlib import Path
from tkinter.font import Font
from typing import Final

import matplotlib

# File paths
H5_PICKLE_PATH: Final[Path] = Path('metadata.pkl')
TIFF_EXPORT_PATH: Final[Path] = Path('export.tif')

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
BEFORE_ODOR_TIME: Final[float] = 3000  # milliseconds
AFTER_ODOR_TIME: Final[float] = 3000  # milliseconds
CALCIUM_VIDEO_DT: Final[float] = 33  # milliseconds

# Plot settings
LATENCY_THRESHOLD: Final[float] = 0.75

# Tkinter format settings
DEFAULT_WINDOW_WIDTH: Final[int] = 1600
DEFAULT_WINDOW_HEIGHT: Final[int] = 850
DEFAULT_WINDOW_SIZE: Final[
    str] = f'{DEFAULT_WINDOW_WIDTH}x{DEFAULT_WINDOW_HEIGHT}'
SMALL_PAD: Final[int] = 10
LARGE_PAD: Final[int] = 30
HEADING_FONT: Final[Font] = ('', 18)

# Miscellaneous
JOB_SCHEDULE_DELAY: Final[int] = 1000
IMAGE_CACHE_MAX_SIZE: Final[int] = 8

# Program parameters (eventually move these into program inputs)
ROI_ZIP_PATH: Final[Path] = Path(
    'data/251003_passive_test2/RoiSet_ET_2025-10-14.zip')
TIFF_FOLDER: Final[Path] = Path('data/251003_passive_test2/TIFFs')
H5_PATH: Final[Path] = Path(
    'data/251003_passive_test2/mouse0953_sess27_D2025_10_3T12_12_5.h5')
ODOR_DICT: Final[OrderedDict[bytes, str]] = {
    b'empty': 'Empty',
    b'ET5': 'Ethyl Tiglate 5',
    b'ET4': 'Ethyl Tiglate 4',
    b'ET3': 'Ethyl Tiglate 3',
    b'ET2': 'Ethyl Tiglate 2',
    b'ET1': 'Ethyl Tiglate 1',
}

# ROI_ZIP_PATH: Final[Path] = Path(
#     'data/sebolddata/GLOM/ROIs/RoiSet_ET_2025-10-02.zip')
# TIFF_FOLDER: Final[Path] = Path('data/sebolddata/GLOM/TIFFs')
# H5_PATH: Final[Path] = Path(
#     'data/sebolddata/GLOM/mouse0953_sess20_D2025_8_18T12_36_43.h5')
# ODOR_DICT: Final[OrderedDict[bytes, str]] = {
#     b'empty': 'Empty',
#     b'EthylTiglate': 'Ethyl Tiglate',
#     b'2MDA': '2MDA'
# }
