from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Self

import numpy as np
import scipy.ndimage
import tifffile
from numpy.typing import NDArray
from PIL import Image, ImageDraw

from config import (BASELINE_AVG_END, BASELINE_AVG_START, CALCIUM_VIDEO_DT,
                    HEAT_CMAP, MAX_DELTA_F, MIN_DELTA_F, NUM_FRAMES_AVG,
                    NUM_FRAMES_BUYIN, PROPORTION_TRIALS_THRESHOLD,
                    TIFF_EXPORT_PATH)
from datatypes import ROIManager, ROIName, StimID

if TYPE_CHECKING:
    from model import ByStimIDModel, ByTrialModel


def _make_delta_f_image(tiff_arr: NDArray[np.float64],
                        median_tiff_arr: NDArray[np.float64],
                        threshold: int = 100) -> NDArray[np.uint8]:
    tiff_arr_c = tiff_arr.copy()
    tiff_arr_c[median_tiff_arr < threshold] = MIN_DELTA_F
    mask = (median_tiff_arr >= threshold)
    tiff_arr_c[mask] = (tiff_arr_c[mask] -
                        median_tiff_arr[mask]) / median_tiff_arr[mask]
    tiff_arr_c = np.clip(tiff_arr_c, MIN_DELTA_F, MAX_DELTA_F)
    tiff_arr_c = (tiff_arr_c - MIN_DELTA_F) / (MAX_DELTA_F - MIN_DELTA_F)
    tiff_arr_c = HEAT_CMAP(tiff_arr_c, bytes=True)
    return tiff_arr_c


def _draw_rois(draw: ImageDraw.Draw, rois: ROIManager,
               rois_focused: list[ROIName]) -> None:
    for roi_name, roi in rois.items():
        x0, y0 = roi['left'], roi['top']
        x1, y1 = x0 + roi['width'], y0 + roi['height']
        if roi_name in rois_focused:
            draw.ellipse([x0, y0, x1, y1], outline='red', width=3)
            draw.rectangle([x0 - 10, y0 - 10, x0 + 10, y0 + 16],
                           fill=(0, 0, 0, 10))
            draw.text([x0 - 10, y0 - 10],
                      str(rois_focused.index(roi_name) + 1),
                      fill='red',
                      font_size=24)
        else:
            draw.ellipse([x0, y0, x1, y1], outline='blue', width=3)


def _load_image(tiff_path: Path) -> NDArray[np.int16]:
    with tifffile.TiffFile(tiff_path) as tf:
        tiff_arr = tf.asarray(range(len(tf.pages)))
    return tiff_arr


class ByTrialCIHandler:

    def make_caption(self: Self, m: ByTrialModel) -> str:
        md = m.metadata
        sc = md.stim_condition
        return (f'Average of {NUM_FRAMES_AVG} frames after odor presentation '
                f'({m.tiff_path.name})\nOdors = '
                f'{sc.odor1_flow} {sc.odor1.value} and '
                f'{sc.odor2_flow} {sc.odor2.value}, '
                f'Target = {md.target.value}, '
                f'Response = {md.response.value}')

    def load_image(self: Self, tiff_path: Path) -> NDArray[np.int16]:
        return _load_image(tiff_path)

    def render_frame(self: Self, m: ByTrialModel, frame: int) -> Image.Image:
        tiff_arr_processed = np.mean(m.tiff_arr[frame:frame + m.temporal_blur +
                                                1],
                                     axis=0)
        tiff_arr_processed = scipy.ndimage.gaussian_filter(
            tiff_arr_processed, sigma=m.spatial_blur)
        if m.median_tiff_arr is not None:
            tiff_arr_processed = _make_delta_f_image(tiff_arr_processed,
                                                     m.median_tiff_arr)
        tiff_image = Image.fromarray(tiff_arr_processed)
        tiff_image_rgb = Image.new('RGB', tiff_image.size)
        tiff_image_rgb.paste(tiff_image)
        draw = ImageDraw.Draw(tiff_image_rgb)
        x0, y0 = tiff_image.width - 1, 0
        draw.rectangle([x0 - 100, y0, x0, y0 + 30], fill=(0, 0, 0, 10))
        draw.text([x0 - 5, y0 + 5],
                  f'{frame+1}/{m.tiff_arr.shape[0]}',
                  fill='red',
                  font_size=24,
                  anchor='rt')
        if not m.hide_rois:
            _draw_rois(draw, m.rois, m.rois_focused)
        return tiff_image_rgb

    def _render_thumbnail(self: Self, frame_times: NDArray[int],
                          odor_time: int, spatial_blur: float,
                          tiff_arr: NDArray[np.int16],
                          median_tiff_arr: NDArray[np.float64] | None,
                          hide_rois: bool, rois: ROIManager,
                          rois_focused: list[ROIName]) -> Image.Image:
        start_frame = np.argmax(frame_times > odor_time) + NUM_FRAMES_BUYIN
        tiff_arr_processed = np.mean(tiff_arr[start_frame:start_frame +
                                              NUM_FRAMES_AVG],
                                     axis=0)
        tiff_arr_processed = scipy.ndimage.gaussian_filter(tiff_arr_processed,
                                                           sigma=spatial_blur)
        if median_tiff_arr is not None:
            tiff_arr_processed = _make_delta_f_image(tiff_arr_processed,
                                                     median_tiff_arr)
        tiff_image = Image.fromarray(tiff_arr_processed)
        tiff_image_rgb = Image.new('RGB', tiff_image.size)
        tiff_image_rgb.paste(tiff_image)
        if not hide_rois:
            draw = ImageDraw.Draw(tiff_image_rgb)
            _draw_rois(draw, rois, rois_focused)
        return tiff_image_rgb

    def render_thumbnail(self: Self, m: ByTrialModel) -> Image.Image:
        md = m.metadata
        return self._render_thumbnail(md.frame_times, md.odor_time,
                                      m.spatial_blur, m.tiff_arr,
                                      m.median_tiff_arr, m.hide_rois, m.rois,
                                      m.rois_focused)

    def render_ci(self: Self, m: ByTrialModel,
                  frame: int | None) -> Image.Image:
        if frame is None:
            return self.render_thumbnail(m)
        else:
            return self.render_frame(m, frame)

    def calc_median_tiff_arr(
            self: Self, tiff_arr: NDArray[np.int16]) -> NDArray[np.float64]:
        return np.median(tiff_arr[BASELINE_AVG_START:BASELINE_AVG_END], axis=0)

    def export_tiff(self: Self, m: ByTrialModel) -> None:
        im_list = []
        for tiff_path, md in zip(m.tiff_files, m.h5_data):
            print(f'Saving {tiff_path}...')
            with tifffile.TiffFile(tiff_path) as tf:
                tiff_arr = tf.asarray(range(len(tf.pages)))
            if m.plot_delta:
                median_tiff_arr = self.calc_median_tiff_arr(tiff_arr)
            else:
                median_tiff_arr = None
            im = self._render_thumbnail(md.frame_times, md.odor_time,
                                        m.spatial_blur, tiff_arr,
                                        median_tiff_arr, True, m.rois,
                                        m.rois_focused)
            im_list.append(im)
        im_list[0].save(TIFF_EXPORT_PATH, append_images=im_list[1:])
        print('Tiff file exported.')


class ByStimIDCIHandler:

    def make_caption(self: Self, m: ByStimIDModel) -> str:
        sc = m.stim_condition
        return (f'Average of {NUM_FRAMES_AVG} frames after odor presentation '
                f'(average of all {m.num_trials_stim_id} stim ID {m.stim_id} '
                f'trials)\nOdors = {sc.odor1_flow} {sc.odor1.value} and '
                f'{sc.odor2_flow} {sc.odor2.value}')

    def load_image(
        self: Self, m: ByStimIDModel, stim_id: StimID | None
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        if stim_id is None:
            stim_id = m.stim_id
        indices = [
            i for i in range(len(m.tiff_files))
            if m.h5_data[i].stim_id == stim_id
        ]
        tiff_arrs = [_load_image(m.tiff_files[i]) for i in indices]
        frame_times_arr = [m.h5_data[i].frame_times for i in indices]
        odor_times_arr = [m.h5_data[i].odor_time for i in indices]

        frame_times_arr_isort = [np.argsort(ts) for ts in frame_times_arr]
        tiff_arrs = [
            f[indices, :, :]
            for f, indices in zip(tiff_arrs, frame_times_arr_isort)
        ]
        frame_times_arr = [
            ts[indices] - t0 for ts, indices, t0 in zip(
                frame_times_arr, frame_times_arr_isort, odor_times_arr)
        ]

        # Choose time points that are present in at least a critical
        # fraction of trials (odor presentation time = 0)
        min_times_arr = [ts[0] for ts in frame_times_arr]
        max_times_arr = [ts[-1] for ts in frame_times_arr]
        start_time = np.percentile(min_times_arr,
                                   PROPORTION_TRIALS_THRESHOLD * 100)
        end_time = np.percentile(max_times_arr,
                                 (1 - PROPORTION_TRIALS_THRESHOLD) * 100)
        time_arr = np.arange(start_time, end_time, CALCIUM_VIDEO_DT)

        # For each time point t, include frames taken between the timestamps
        # t and t + DT
        image_size = tiff_arrs[0].shape[1:]
        tiff_arr = np.zeros((time_arr.size, *image_size))
        count_arr = np.zeros_like(time_arr)
        for i in range(len(tiff_arrs)):
            arr = tiff_arrs[i]
            frame_times = frame_times_arr[i]
            time_j = 0
            frame_j = 0
            while frame_j < arr.shape[0] and time_j < time_arr.shape[0]:
                if time_arr[time_j] < frame_times[frame_j]:
                    if frame_times[
                            frame_j] < time_arr[time_j] + CALCIUM_VIDEO_DT:
                        tiff_arr[time_j] = tiff_arr[time_j] + arr[frame_j]
                        count_arr[time_j] = count_arr[time_j] + 1
                        frame_j = frame_j + 1
                        time_j = time_j + 1
                    else:
                        time_j = time_j + 1
                else:
                    frame_j = frame_j + 1
        avg_tiff_arr = tiff_arr / count_arr[:, np.newaxis, np.newaxis]
        return avg_tiff_arr, time_arr

    def render_frame(self: Self, m: ByStimIDModel, frame: int) -> Image.Image:
        tiff_arr_processed = np.mean(m.tiff_arr[frame:frame + m.temporal_blur +
                                                1],
                                     axis=0)
        tiff_arr_processed = scipy.ndimage.gaussian_filter(
            tiff_arr_processed, sigma=m.spatial_blur)
        if m.median_tiff_arr is not None:
            tiff_arr_processed = _make_delta_f_image(tiff_arr_processed,
                                                     m.median_tiff_arr)
        tiff_image = Image.fromarray(tiff_arr_processed)
        tiff_image_rgb = Image.new('RGB', tiff_image.size)
        tiff_image_rgb.paste(tiff_image)
        draw = ImageDraw.Draw(tiff_image_rgb)
        x0, y0 = tiff_image.width - 1, 0
        draw.rectangle([x0 - 100, y0, x0, y0 + 30], fill=(0, 0, 0, 10))
        draw.text([x0 - 5, y0 + 5],
                  f'{frame+1}/{m.tiff_arr.shape[0]}',
                  fill='red',
                  font_size=24,
                  anchor='rt')
        if not m.hide_rois:
            _draw_rois(draw, m.rois, m.rois_focused)
        return tiff_image_rgb

    def _render_thumbnail(self: Self, spatial_blur: float,
                          tiff_arr: NDArray[np.float64],
                          time_arr: NDArray[np.float64],
                          median_tiff_arr: NDArray[np.float64] | None,
                          hide_rois: bool, rois: ROIManager,
                          rois_focused: list[ROIName]) -> Image.Image:
        start_frame = np.argmax(time_arr > 0) + NUM_FRAMES_BUYIN
        tiff_arr_processed = np.mean(tiff_arr[start_frame:start_frame +
                                              NUM_FRAMES_AVG],
                                     axis=0)
        tiff_arr_processed = scipy.ndimage.gaussian_filter(tiff_arr_processed,
                                                           sigma=spatial_blur)
        if median_tiff_arr is not None:
            tiff_arr_processed = _make_delta_f_image(tiff_arr_processed,
                                                     median_tiff_arr)
        tiff_image = Image.fromarray(tiff_arr_processed)
        tiff_image_rgb = Image.new('RGB', tiff_image.size)
        tiff_image_rgb.paste(tiff_image)
        if not hide_rois:
            draw = ImageDraw.Draw(tiff_image_rgb)
            _draw_rois(draw, rois, rois_focused)
        return tiff_image_rgb

    def render_thumbnail(self: Self, m: ByStimIDModel) -> Image.Image:
        return self._render_thumbnail(m.spatial_blur, m.tiff_arr, m.time_arr,
                                      m.median_tiff_arr, m.hide_rois, m.rois,
                                      m.rois_focused)

    def render_ci(self: Self, m: ByStimIDModel,
                  frame: int | None) -> Image.Image:
        if frame is None:
            return self.render_thumbnail(m)
        else:
            return self.render_frame(m, frame)

    def calc_median_tiff_arr(
            self: Self, tiff_arr: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.median(tiff_arr[BASELINE_AVG_START:BASELINE_AVG_END], axis=0)

    def export_tiff(self: Self, m: ByStimIDModel) -> None:
        im_list = []
        for i in range(len(m.stim_ids)):
            print(f'Saving Stim ID {m.stim_ids[i]}...')
            tiff_arr, time_arr = m.load_tiff_arr(i)
            if m.plot_delta:
                median_tiff_arr = self.calc_median_tiff_arr(tiff_arr)
            else:
                median_tiff_arr = None
            im = self._render_thumbnail(m.spatial_blur, tiff_arr, time_arr,
                                        median_tiff_arr, True, m.rois,
                                        m.rois_focused)
            im_list.append(im)
        im_list[0].save(TIFF_EXPORT_PATH, append_images=im_list[1:])
        print('Tiff file exported.')
