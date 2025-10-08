from pathlib import Path
from typing import Self

import numpy as np
from numpy.typing import NDArray

import calcium_image
from datatypes import (GUIStateMessage, PlotSetting, ROIManager, ROIName,
                       TrialMetadata, VariableSet)


class GUIState:

    def __init__(self: Self, tif_files: list[Path],
                 h5_data: list[TrialMetadata], roi_zip_path: Path,
                 variable_set: VariableSet) -> None:
        self._tif_files = tif_files
        self._tif_file_i = 0
        self._tif_path = tif_files[self._tif_file_i]
        self._tiff_arr = calcium_image.load_image(self._tif_path)
        self._median_tiff_arr = None
        self._h5_data = h5_data
        self._metadata = h5_data[self._tif_file_i]
        self._rois = ROIManager.from_zip(roi_zip_path)
        self._rois_focused = []
        self._image = None
        self._window_width = 0
        self._window_height = 0
        self._variable_set = variable_set

    def get_tif_path(self: Self) -> Path:
        return self._tif_path

    def get_tif_file_i(self: Self) -> int:
        return self._tif_file_i

    def set_tif_file_i(self: Self, i: int) -> None:
        self._tif_file_i = i % len(self._tif_files)
        self._tif_path = self._tif_files[self._tif_file_i]
        self._tiff_arr = calcium_image.load_image(self._tif_path)
        self._metadata = self._h5_data[self._tif_file_i]

    def get_tiff_arr(self: Self) -> NDArray[np.int16]:
        return self._tiff_arr

    def get_median_tiff_arr(self: Self) -> NDArray[np.float64]:
        if self._median_tiff_arr is None:
            self._median_tiff_arr = calcium_image.calc_median_tiff_arr(
                self._tiff_arr)
        return self._median_tiff_arr

    def get_metadata(self: Self) -> TrialMetadata:
        return self._metadata

    def get_all_rois(self: Self) -> ROIManager:
        return self._rois

    def get_rois_focused(self: Self) -> list[ROIName]:
        return self._rois_focused

    def set_rois_focused(self: Self, r: list[ROIName]) -> None:
        self._rois_focused = r

    def toggle_roi(self: Self, roi_name: str) -> None:
        if roi_name in self._rois_focused:
            self._rois_focused.remove(roi_name)
        else:
            self._rois_focused.append(roi_name)

    def was_window_resized(self: Self, width: int, height: int) -> bool:
        return self._window_width != width or self._window_height != height

    def set_window_size(self: Self, width: int, height: int) -> None:
        self._window_width = width
        self._window_height = height

    def get_variable_set(self: Self) -> VariableSet:
        return self._variable_set

    def make_message(self: Self) -> GUIStateMessage:
        plot_delta = self.get_variable_set().plot_delta_var.get()
        median_tiff_arr = self.get_median_tiff_arr() if plot_delta else None
        return GUIStateMessage(
            tif_path=self.get_tif_path(),
            tiff_arr=self.get_tiff_arr(),
            metadata=self.get_metadata(),
            rois=self.get_all_rois(),
            rois_focused=self.get_rois_focused(),
            hide_rois=self.get_variable_set().hide_rois_var.get(),
            plot_delta=plot_delta,
            median_tiff_arr=median_tiff_arr,
            spatial_blur=self.get_variable_set().spatial_blur_var.get(),
            temporal_blur=self.get_variable_set().temporal_blur_var.get(),
            plot_setting=PlotSetting(
                self.get_variable_set().plot_setting_var.get()))
