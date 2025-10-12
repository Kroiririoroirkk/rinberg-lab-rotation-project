import pickle
from pathlib import Path
from typing import Self

import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from PIL import Image

from ci_handler import ByStimIDCIHandler, ByTrialCIHandler
from config import DEFAULT_SPATIAL_BLUR, DEFAULT_TEMPORAL_BLUR
from datatypes import (ByStimIDPlotSetting, ByTrialPlotSetting, PageSetting,
                       ROIManager, StimCondition, StimID, TrialMetadata,
                       for_page)
from plot_handler import ByStimIDPlotHandler, ByTrialPlotHandler

PAGE_SETTING_MODEL_DICT: dict[PageSetting, type] = dict()


@for_page(PAGE_SETTING_MODEL_DICT, PageSetting.BY_TRIAL)
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
        self.plot_setting = ByTrialPlotSetting.NONE
        self.ci_handler = ByTrialCIHandler()
        self.plot_handler = ByTrialPlotHandler()

    @property
    def tiff_files(self: Self) -> list[Path]:
        return self.parent.tiff_files

    @property
    def h5_data(self: Self) -> list[TrialMetadata]:
        return self.parent.h5_data

    @property
    def stim_condition_dict(self: Self) -> dict[StimID, StimCondition]:
        return self.parent.stim_condition_dict

    @property
    def rois(self: Self) -> ROIManager:
        return self.parent.rois

    @property
    def page_setting(self: Self) -> PageSetting:
        return self.parent.page_setting

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

    def export_tiff(self: Self) -> None:
        self.ci_handler.export_tif(self)

    def delete_running_lines(self: Self) -> None:
        self.plot_handler.delete_running_lines()

    def update_ci(self: Self, i: int | None) -> tuple[str, Image.Image]:
        return (self.ci_handler.make_caption(self),
                self.ci_handler.render_ci(self, i))

    def update_plot(self: Self, fig: Figure, i: int | None) -> None:
        self.plot_handler.render_plot(fig, self, i)


@for_page(PAGE_SETTING_MODEL_DICT, PageSetting.BY_STIM_ID)
class ByStimIDModel:

    def __init__(self: Self, parent: 'Model') -> None:
        self.parent = parent
        self.stim_ids = sorted(list(self.stim_condition_dict.keys()))
        self._stim_id_i = 0
        self._median_tiff_arr = None
        self._tiff_arrs = dict()
        self._time_arrs = dict()
        self.rois_focused = []
        self.start_from_odor = True
        self.hide_rois = False
        self.plot_delta = True
        self.spatial_blur = DEFAULT_SPATIAL_BLUR
        self.temporal_blur = DEFAULT_TEMPORAL_BLUR
        self.plot_setting = ByStimIDPlotSetting.NONE
        self.ci_handler = ByStimIDCIHandler()
        self.plot_handler = ByStimIDPlotHandler()

    def get_stim_ids(self: Self) -> list[StimID]:
        stim_ids = []
        for metadata in self.h5_data:
            if metadata.stim_id not in stim_ids:
                stim_ids.append(metadata.stim_id)
        return sorted(stim_ids)

    @property
    def tiff_files(self: Self) -> list[Path]:
        return self.parent.tiff_files

    @property
    def h5_data(self: Self) -> list[TrialMetadata]:
        return self.parent.h5_data

    @property
    def stim_condition_dict(self: Self) -> dict[StimID, StimCondition]:
        return self.parent.stim_condition_dict

    @property
    def rois(self: Self) -> ROIManager:
        return self.parent.rois

    @property
    def page_setting(self: Self) -> PageSetting:
        return self.parent.page_setting

    @property
    def stim_id_i(self: Self) -> int:
        return self._stim_id_i

    @stim_id_i.setter
    def stim_id_i(self: Self, i: int) -> None:
        self._stim_id_i = i % len(self.stim_ids)
        self._tiff_arr = None

    @property
    def stim_id(self: Self) -> StimID:
        return self.stim_ids[self.stim_id_i]

    @property
    def stim_condition(self: Self) -> StimCondition:
        return self.stim_condition_dict[self.stim_id]

    @property
    def tiff_arr(self: Self) -> NDArray[np.int16]:
        i = self.stim_id_i
        if i not in self._tiff_arrs:
            loaded = self.ci_handler.load_image(self)
            self._tiff_arrs[i], self._time_arrs[i] = loaded
        return self._tiff_arrs[i]

    @property
    def time_arr(self: Self) -> NDArray[np.int16]:
        i = self.stim_id_i
        if i not in self._tiff_arrs:
            img = self.ci_handler.load_image(self)
            self._tiff_arrs[i], self._time_arrs[i] = img
        return self._time_arrs[i]

    @property
    def median_tiff_arr(self: Self) -> NDArray[np.float64]:
        if not self.plot_delta:
            return None
        if self._median_tiff_arr is None:
            self._median_tiff_arr = self.ci_handler.calc_median_tiff_arr(
                self.tiff_arr)
        return self._median_tiff_arr

    def toggle_roi(self: Self, roi_name: str) -> None:
        if roi_name in self.rois_focused:
            self.rois_focused.remove(roi_name)
        else:
            self.rois_focused.append(roi_name)

    def export_tiff(self: Self) -> None:
        self.ci_handler.export_tiff(self)

    def delete_running_lines(self: Self) -> None:
        self.plot_handler.delete_running_lines()

    def update_ci(self: Self, i: int | None) -> tuple[str, Image.Image]:
        return (self.ci_handler.make_caption(self),
                self.ci_handler.render_ci(self, i))

    def update_plot(self: Self, fig: Figure, i: int | None) -> None:
        self.plot_handler.render_plot(fig, self, i)


class Model:

    def __init__(self: Self, tiff_folder_path: Path, h5_path: Path,
                 h5_pickle_path: Path, roi_zip_path: Path):
        self.tiff_files = self.load_tiff_file_paths(tiff_folder_path)
        self.h5_data, self.stim_condition_dict = self.load_h5(
            h5_path, h5_pickle_path)
        self.rois = ROIManager.from_zip(roi_zip_path)
        self.page_setting = PageSetting.BY_TRIAL
        self.children = dict()
        self.create_children()

    def load_tiff_file_paths(self: Self, tiff_folder_path: Path) -> list[Path]:
        return sorted(
            [f for f in tiff_folder_path.iterdir() if f.suffix == '.tif'])

    def load_h5(
        self: Self, h5_path: Path, h5_pickle_path: Path
    ) -> tuple[list[TrialMetadata], dict[StimID, StimCondition]]:
        print('Loading H5 file...')
        try:
            with h5_pickle_path.open('rb') as f:
                h5_data, stim_condition_dict = pickle.load(f)
        except FileNotFoundError:
            h5_data, stim_condition_dict = TrialMetadata.list_from_h5(h5_path)
            with h5_pickle_path.open('wb') as f:
                pickle.dump((h5_data, stim_condition_dict), f)
        return h5_data, stim_condition_dict

    def create_children(self: Self) -> None:
        for s in PageSetting:
            self.children[s] = PAGE_SETTING_MODEL_DICT[s](self)
