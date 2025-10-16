from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Self

import numpy as np
import scipy.ndimage
import tifffile
from numpy.typing import NDArray
from PIL import Image, ImageDraw

from config import (AFTER_ODOR_TIME, BASELINE_AVG_END, BASELINE_AVG_START,
                    BEFORE_ODOR_TIME, CALCIUM_VIDEO_DT, HEAT_CMAP, MAX_DELTA_F,
                    MIN_DELTA_F, NUM_FRAMES_AVG, NUM_FRAMES_BUYIN,
                    TIFF_EXPORT_PATH)
from datatypes import Odor, ROIManager, ROIName, StimID

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
        # Assume PID readings are for the odor 2 port
        key = (sc.odor2, int(sc.odor2_flow))
        if key in m.pid_reading_dict and sc.odor1 == Odor.EMPTY:
            return (f'Average of {NUM_FRAMES_AVG} frames after odor '
                    f'presentation ({m.tiff_path.name})\nOdors = '
                    f'{sc.odor1_flow} {sc.odor1} and '
                    f'{sc.odor2_flow} {sc.odor2} '
                    f'(stim condition {md.stim_id}, est. PID reading '
                    f'{m.pid_reading_dict[key]:.2e})\n'
                    f'Target = {md.target.value}, '
                    f'Response = {md.response.value}')
        else:
            return (f'Average of {NUM_FRAMES_AVG} frames after odor '
                    f'presentation ({m.tiff_path.name})\nOdors = '
                    f'{sc.odor1_flow} {sc.odor1} and '
                    f'{sc.odor2_flow} {sc.odor2} '
                    f'(stim condition {md.stim_id})\n'
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
            if isinstance(md, str):
                print(f'Skipping {tiff_path} because of invalid metadata...')
                continue
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
        # Assume PID readings are for the odor 2 port
        key = (sc.odor2, int(sc.odor2_flow))
        if key in m.pid_reading_dict and sc.odor1 == Odor.EMPTY:
            return (f'Average of {NUM_FRAMES_AVG} frames after odor '
                    f'presentation (average of all {m.num_trials_stim_id} '
                    f'stim ID {m.stim_id} trials)\n'
                    f'Odors = {sc.odor1_flow} {sc.odor1} and '
                    f'{sc.odor2_flow} {sc.odor2} '
                    f'(est. PID reading {m.pid_reading_dict[key]:.2e})')
        else:
            return (f'Average of {NUM_FRAMES_AVG} frames after odor '
                    f'presentation (average of all {m.num_trials_stim_id} '
                    f'stim ID {m.stim_id} trials)\n'
                    f'Odors = {sc.odor1_flow} {sc.odor1} and '
                    f'{sc.odor2_flow} {sc.odor2}')

    def load_image(
        self: Self,
        m: ByStimIDModel,
        stim_id: StimID | None,
        return_individual_images: bool = False
    ) -> tuple[NDArray[np.float64], NDArray[np.float64],
               NDArray[np.int64]] | tuple[
                   NDArray[np.float64], NDArray[np.float64], NDArray[np.int64],
                   list[NDArray[np.float64]]]:
        if stim_id is None:
            stim_id = m.stim_id
        indices = [
            i for i in range(len(m.tiff_files))
            if not isinstance(m.h5_data[i], str)
            and m.h5_data[i].stim_id == stim_id
        ]
        tiff_arrs = [_load_image(m.tiff_files[i]) for i in indices]
        frame_times_arr = [m.h5_data[i].frame_times for i in indices]
        odor_times_arr = [m.h5_data[i].odor_time for i in indices]

        tiff_arrs, frame_tims_arr = list(
            zip(*[
                m.parent.match_frames_with_timestamps(arr, frame_times)
                for arr, frame_times in zip(tiff_arrs, frame_times_arr)
            ]))

        frame_times_arr_isort = [np.argsort(ts) for ts in frame_times_arr]
        tiff_arrs = [
            f[indices, :, :]
            for f, indices in zip(tiff_arrs, frame_times_arr_isort)
        ]
        frame_times_arr = [
            ts[indices] - t0 for ts, indices, t0 in zip(
                frame_times_arr, frame_times_arr_isort, odor_times_arr)
        ]

        start_time = -BEFORE_ODOR_TIME
        end_time = AFTER_ODOR_TIME
        dt = CALCIUM_VIDEO_DT
        time_arr = np.arange(start_time, end_time, dt)

        def interpolate(
                tiff_arr_trial: NDArray[np.int16],
                time_arr_trial: NDArray[np.int64]) -> NDArray[np.float64]:
            # Remove duplicate time entries from time_arr_trial by deleting
            # all frames after the first
            time_arr_unique, unique_idx = np.unique(time_arr_trial,
                                                    return_index=True)
            tiff_arr_unique = tiff_arr_trial[unique_idx]

            # Perform linear interpolation, use vectorization to be faster than
            # scipy.interpolate.make_interp_spline implementation
            idx = np.searchsorted(time_arr_unique, time_arr, side='left')
            idx = np.clip(idx, 1, time_arr_unique.size - 1)
            x0 = time_arr_unique[idx - 1]
            x1 = time_arr_unique[idx]
            w = (time_arr - x0) / (x1 - x0)
            y0 = tiff_arr_unique[idx - 1]
            y1 = tiff_arr_unique[idx]
            tiff_arr = (1 - w)[:, np.newaxis,
                               np.newaxis] * y0 + w[:, np.newaxis,
                                                    np.newaxis] * y1
            return tiff_arr

        if return_individual_images:
            trial_ids = []
            interpolated_tiff_arrs = []
            for i, tiff_arr_trial, time_arr_trial in zip(
                    indices, tiff_arrs, frame_times_arr):
                if time_arr_trial[0] > start_time:
                    continue
                if time_arr_trial[-1] < end_time:
                    continue
                trial_ids.append(i)
                tiff_arr = interpolate(tiff_arr_trial, time_arr_trial)
                interpolated_tiff_arrs.append(tiff_arr)
            trial_ids = np.array(trial_ids, dtype=np.int64)
            avg_tiff_arr = np.mean(interpolated_tiff_arrs, axis=0)
            return avg_tiff_arr, time_arr, trial_ids, interpolated_tiff_arrs
        else:
            avg_tiff_arr = np.zeros((time_arr.size, *tiff_arrs[0].shape[1:]))
            trial_ids = []
            for i, tiff_arr_trial, time_arr_trial in zip(
                    indices, tiff_arrs, frame_times_arr):
                if time_arr_trial[0] > start_time:
                    continue
                if time_arr_trial[-1] < end_time:
                    continue
                trial_ids.append(i)
                tiff_arr = interpolate(tiff_arr_trial, time_arr_trial)
                avg_tiff_arr += tiff_arr
            trial_ids = np.array(trial_ids, dtype=np.int64)
            avg_tiff_arr /= trial_ids.size
            return avg_tiff_arr, time_arr, trial_ids

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
            tiff_arr, time_arr, _ = m.load_tiff_arr(i)
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
