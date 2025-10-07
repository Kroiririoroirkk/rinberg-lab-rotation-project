from collections import OrderedDict
from pathlib import Path

import numpy as np
import scipy.ndimage
import tifffile
from numpy.typing import NDArray
from PIL import Image, ImageDraw

from config import (BASELINE_AVG_END, BASELINE_AVG_START, HEAT_CMAP,
                    MAX_DELTA_F, MIN_DELTA_F, NUM_FRAMES_AVG, NUM_FRAMES_BUYIN,
                    TIF_EXPORT_PATH)
from datatypes import ROIManager, ROIName, TrialMetadata


def make_caption(tif_path: Path, trial_metadata: TrialMetadata) -> str:
    return (f'Average of {NUM_FRAMES_AVG} frames after odor presentation '
            f'({tif_path.name})\nOdors = '
            f'{trial_metadata.odor1_flow}% {trial_metadata.odor1.value}, '
            f'{trial_metadata.odor2_flow}% {trial_metadata.odor2.value}\n'
            f'Target = {trial_metadata.target.value}, '
            f'Response = {trial_metadata.response.value}')


def load_image(tif_path: Path) -> NDArray[np.int16]:
    with tifffile.TiffFile(tif_path) as tf:
        tiff_arr = tf.asarray(range(len(tf.pages)))
    return tiff_arr


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


def render_frame(tiff_arr: NDArray[np.int16], rois: ROIManager,
                 rois_focused: list[ROIName], hide_rois: bool, frame: int,
                 spatial_blur: int, temporal_blur: int,
                 median_tiff_arr: NDArray[np.int16] | None) -> Image.Image:
    tiff_arr_processed = np.mean(tiff_arr[frame:frame + temporal_blur + 1],
                                 axis=0)
    tiff_arr_processed = scipy.ndimage.gaussian_filter(tiff_arr_processed,
                                                       sigma=spatial_blur)
    if median_tiff_arr is not None:
        tiff_arr_processed = _make_delta_f_image(tiff_arr_processed,
                                                 median_tiff_arr)
    tiff_image = Image.fromarray(tiff_arr_processed)
    tiff_image_rgb = Image.new('RGB', tiff_image.size)
    tiff_image_rgb.paste(tiff_image)
    draw = ImageDraw.Draw(tiff_image_rgb)
    x0, y0 = tiff_image.width - 1, 0
    draw.rectangle([x0 - 100, y0, x0, y0 + 30], fill=(0, 0, 0, 10))
    draw.text([x0 - 5, y0 + 5],
              f'{frame+1}/{tiff_arr.shape[0]}',
              fill='red',
              font_size=24,
              anchor='rt')
    if not hide_rois:
        _draw_rois(draw, rois, rois_focused)
    return tiff_image_rgb


def render_thumbnail(
        tiff_arr: NDArray[np.int16], metadata: TrialMetadata, rois: ROIManager,
        rois_focused: list[ROIName], hide_rois: bool, spatial_blur: float,
        median_tiff_arr: NDArray[np.float64] | None) -> Image.Image:
    start_frame = np.argmax(
        metadata.frame_times > metadata.odor_time) + NUM_FRAMES_BUYIN
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


def calc_median_tiff_arr(tiff_arr: NDArray[np.int16]) -> NDArray[np.int16]:
    return np.median(tiff_arr[BASELINE_AVG_START:BASELINE_AVG_END], axis=0)


def export_tif(tif_files: list[Path], h5_data: list[TrialMetadata],
               spatial_blur: float, plot_delta: bool) -> None:
    im_list = []
    for tif_path, metadata in zip(tif_files, h5_data):
        print(tif_path)
        with tifffile.TiffFile(tif_path) as tf:
            tiff_arr = tf.asarray(range(len(tf.pages)))
        if plot_delta:
            median_tiff_arr = calc_median_tiff_arr(tiff_arr)
        else:
            median_tiff_arr = None
        im = render_thumbnail(tiff_arr, metadata, ROIManager(OrderedDict()),
                              [], True, spatial_blur, median_tiff_arr)
        im_list.append(im)
    im_list[0].save(TIF_EXPORT_PATH, append_images=im_list[1:])
