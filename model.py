import pickle
from pathlib import Path
from typing import Self

import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from PIL import Image

from ci_handler import CIHandler
from config import DEFAULT_SPATIAL_BLUR, DEFAULT_TEMPORAL_BLUR
from datatypes import (GUIStateMessage, PageSetting, PlotSetting, ROIManager,
                       TrialMetadata)
from plot_handler import PlotHandler


class ByTrialModel:

    def __init__(self: Self, parent: 'Model') -> None:
        self.parent = parent
        self._tiff_file_i = 0
        self._median_tiff_arr = None
        self._tiff_arr = None
        self.rois_focused = []
        self.start_from_odor = True
        self.hide_rois = False
        self.plot_delta = True
        self.spatial_blur = DEFAULT_SPATIAL_BLUR
        self.temporal_blur = DEFAULT_TEMPORAL_BLUR
        self.plot_setting = PlotSetting.NONE
        self.ci_handler = CIHandler()
        self.plot_handler = PlotHandler()

    @property
    def tiff_files(self: Self) -> list[Path]:
        return self.parent.tiff_files

    @property
    def h5_data(self: Self) -> list[TrialMetadata]:
        return self.parent.h5_data

    @property
    def rois(self: Self) -> ROIManager:
        return self.parent.rois

    @property
    def tiff_file_i(self: Self) -> int:
        return self._tiff_file_i

    @tiff_file_i.setter
    def tiff_file_i(self: Self, i: int) -> None:
        self._tiff_file_i = i % len(self.tiff_files)
        self._tiff_arr = None

    @property
    def tiff_path(self: Self) -> Path:
        return self.tiff_files[self.tiff_file_i]

    @property
    def tiff_arr(self: Self) -> NDArray[np.int16]:
        if self._tiff_arr is None:
            self._tiff_arr = self.ci_handler.load_image(self.tiff_path)
        return self._tiff_arr

    @property
    def median_tiff_arr(self: Self) -> NDArray[np.float64]:
        if not self.plot_delta:
            return None
        if self._median_tiff_arr is None:
            self._median_tiff_arr = self.ci_handler.calc_median_tiff_arr(
                self.tiff_arr)
        return self._median_tiff_arr

    @property
    def metadata(self: Self) -> TrialMetadata:
        return self.h5_data[self.tiff_file_i]

    def toggle_roi(self: Self, roi_name: str) -> None:
        if roi_name in self.rois_focused:
            self.rois_focused.remove(roi_name)
        else:
            self.rois_focused.append(roi_name)

    def make_message(self: Self) -> GUIStateMessage:
        return GUIStateMessage(tiff_path=self.tiff_path,
                               tiff_arr=self.tiff_arr,
                               metadata=self.metadata,
                               rois=self.rois,
                               rois_focused=self.rois_focused,
                               hide_rois=self.hide_rois,
                               plot_delta=self.plot_delta,
                               median_tiff_arr=self.median_tiff_arr,
                               spatial_blur=self.spatial_blur,
                               temporal_blur=self.temporal_blur,
                               plot_setting=self.plot_setting)

    def export_tiff(self: Self) -> None:
        self.ci_handler.export_tif(self.tiff_files, self.h5_data,
                                   self.make_message())

    def delete_running_lines(self: Self) -> None:
        self.plot_handler.delete_running_lines()

    def update_ci(self: Self, i: int | None) -> tuple[str, Image.Image]:
        message = self.make_message()
        return (self.ci_handler.make_caption(message),
                self.ci_handler.render_ci(message, i))

    def update_plot(self: Self, fig: Figure, i: int | None) -> None:
        message = self.make_message()
        self.plot_handler.render_plot(fig, message, i)


class ByStimTypeModel:

    def __init__(self: Self, parent: 'Model') -> None:
        self.parent = parent


PAGE_SETTING_MODEL_DICT = {
    PageSetting.BY_TRIAL: ByTrialModel,
    PageSetting.BY_STIM_TYPE: ByStimTypeModel
}


class Model:

    def __init__(self: Self, tiff_folder_path: Path, h5_path: Path,
                 h5_pickle_path: Path, roi_zip_path: Path):
        self.tiff_files = self.load_tiff_file_paths(tiff_folder_path)
        self.h5_data = self.load_h5(h5_path, h5_pickle_path)
        self.rois = ROIManager.from_zip(roi_zip_path)
        self.children = dict()
        self.create_children()

    def load_tiff_file_paths(self: Self, tiff_folder_path: Path) -> list[Path]:
        return sorted(
            [f for f in tiff_folder_path.iterdir() if f.suffix == '.tif'])

    def load_h5(self: Self, h5_path: Path,
                h5_pickle_path: Path) -> list[TrialMetadata]:
        print('Loading H5 file...')
        try:
            with h5_pickle_path.open('rb') as f:
                h5_data = pickle.load(f)
        except FileNotFoundError:
            h5_data = TrialMetadata.list_from_h5(h5_path)
            with h5_pickle_path.open('wb') as f:
                pickle.dump(h5_data, f)
        return h5_data

    def create_children(self: Self) -> None:
        for s in PageSetting:
            self.children[s] = PAGE_SETTING_MODEL_DICT[s](self)
