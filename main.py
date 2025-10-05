"""Calcium Imaging Analyzer

The purpose of this script is to allow for the convenient viewing and analysis
of two-photon calcium imaging data, together with behavioral (two-alternative
forced choice task responses) and physiological data (respiration patterns).

Author: Eric Tao (Eric.Tao@nyulangone.org)
Date created: 2025-09-23
Date last updated: 2025-09-30
"""

import pickle
import tkinter as tk
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from tkinter import ttk
from tkinter.font import Font
from typing import Final, Self

import h5py
import numpy as np
import read_roi
import tifffile
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)
from matplotlib.figure import Figure
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageTk

ROI_ZIP_PATH: Final[Path] = Path('data/GLOM/ROIs/RoiSet_ET_2025-10-02.zip')
TIF_FOLDER: Final[Path] = Path('data/GLOM/TIFFs')
H5_PATH: Final[Path] = Path(
    'data/GLOM/mouse0953_sess20_D2025_8_18T12_36_43.h5')
H5_PICKLE_PATH: Final[Path] = Path('metadata.pkl')
FRAMES_AVG_START: Final[int] = 0
FRAMES_AVG_END: Final[int] = 100
BASELINE_AVG_START: Final[int] = 0
BASELINE_AVG_END: Final[int] = 100
DEFAULT_WINDOW_SIZE: Final[str] = '1600x850'
SMALL_PAD: int = 10
MEDIUM_PAD: int = 20
LARGE_PAD: int = 30
HEADING_FONT: Font = ('', 18)
SMOOTHING_DISTANCE: int = 5


class Odor(Enum):
    ETHYL_TIGLATE: Self = 'Ethyl Tiglate'
    TMBA: Self = '2MBA'
    EMPTY: Self = 'Empty'

    @staticmethod
    def lookup(s: bytes) -> Self | None:
        d = {
            b'EthylTiglate': Odor.ETHYL_TIGLATE,
            b'2MBA': Odor.TMBA,
            b'empty': Odor.EMPTY
        }
        return d.get(s)


class Choice(Enum):
    LEFT: Self = 'Left'
    RIGHT: Self = 'Right'
    MISS: Self = 'Miss'

    @staticmethod
    def lookup_target(i: int) -> Self | None:
        d = {0: Choice.LEFT, 1: Choice.RIGHT}
        return d.get(i)

    @staticmethod
    def lookup_response(i: int) -> Self | None:
        d = {
            1: Choice.RIGHT,
            2: Choice.LEFT,
            3: Choice.LEFT,
            4: Choice.RIGHT,
            5: Choice.MISS,
            6: Choice.MISS
        }
        return d.get(i)


@dataclass
class TrialMetadata:
    frame_times: NDArray[int]  # in ms
    odor_time: int  # in ms
    lick1_times: NDArray[int]  # in ms
    lick2_times: NDArray[int]  # in ms
    sniff_times: NDArray[int]  # in ms
    sniff_values: NDArray[int]  # in relative units
    odor1: Odor
    odor2: Odor
    odor1_flow: float
    odor2_flow: float
    target: Choice
    response: Choice
    correct: bool


ROIName = str
ROI = dict[str, str | int]


class PlotSetting(Enum):
    NONE: Self = 'None'
    BEHAVIOR: Self = 'Behavior'
    FLUORESCENCE: Self = 'Fluorescence'


def read_clumped_arr(arr: h5py._hl.dataset.Dataset) -> list[int]:
    return np.array([int(x) for x in np.concatenate(arr)
                     ]) if arr.size else np.array([])


def load_h5() -> list[TrialMetadata]:
    trial_metadata_list = []
    f = h5py.File(H5_PATH, 'r')
    all_trials_f = f['Trials']
    recorded_trials = np.nonzero(all_trials_f['record'])[0]
    for i in recorded_trials:
        trial_f = f[f'Trial{i+1:04d}']
        frame_times = read_clumped_arr(trial_f['frame_triggers'])
        odor_time = int(all_trials_f['fvOnTime_1st'][i])
        lick1_times = read_clumped_arr(trial_f['lick1'])
        lick2_times = read_clumped_arr(trial_f['lick2'])
        sniff_chunk_len1 = [len(s) for s in trial_f['sniff']]
        sniff_chunk_len2 = [int(s) for s in trial_f['Events']['sniff_samples']]
        sniff_chunk_len2 = [s for s in sniff_chunk_len2 if s != 0]
        if sniff_chunk_len1 != sniff_chunk_len2:
            raise ValueError(
                f'Sniff chunk lengths not consistent for Trial{i+1:04d}: '
                f'{sniff_chunk_len1} != {sniff_chunk_len2}.')
        sniff_times = []
        for j in range(trial_f['Events'].shape[0] - 1):
            chunk_start_time, chunk_size = trial_f['Events'][j]
            chunk_end_time, _ = trial_f['Events'][j + 1]
            sniff_times.extend([
                int(x) for x in np.linspace(chunk_start_time,
                                            chunk_end_time,
                                            chunk_size,
                                            endpoint=False)
            ])
        sniff_times = np.array(sniff_times)
        sniff_values = read_clumped_arr(trial_f['sniff'][:-1])
        if sniff_times.size != sniff_values.size:
            raise ValueError(
                'Sniff time and sniff value lists do not match up.')
        o1, o2 = all_trials_f['olfa_1st_0_odor'][i], all_trials_f[
            'olfa_1st_1_odor'][i]
        odor1, odor2 = Odor.lookup(o1), Odor.lookup(o2)
        if odor1 is None or odor2 is None:
            raise ValueError(f'Odor lookup failed on odors: {o1}, {o2}.')
        odor1_flow = float(all_trials_f['olfa_1st_0_mfc_1_flow'][i])
        odor2_flow = float(all_trials_f['olfa_1st_1_mfc_1_flow'][i])
        t = all_trials_f['trialtype'][i]
        target = Choice.lookup_target(t)
        if target is None:
            raise ValueError(f'Target lookup failed on input: {t}.')
        r = all_trials_f['result'][i]
        response = Choice.lookup_response(r)
        if response is None:
            raise ValueError(f'Response lookup failed on input: {r}.')
        correct = (target == response)
        trial_metadata_list.append(
            TrialMetadata(frame_times=frame_times,
                          odor_time=odor_time,
                          lick1_times=lick1_times,
                          lick2_times=lick2_times,
                          sniff_times=sniff_times,
                          sniff_values=sniff_values,
                          odor1=odor1,
                          odor2=odor2,
                          odor1_flow=odor1_flow,
                          odor2_flow=odor2_flow,
                          target=target,
                          response=response,
                          correct=correct))
    return trial_metadata_list


def load_rois() -> OrderedDict[ROIName, ROI]:
    rois = read_roi.read_roi_zip(ROI_ZIP_PATH)
    for v in rois.values():
        if v.get('type') != 'oval':
            raise ValueError('Cannot process non-oval ROI.')
    return rois


def load_caption(tif_path: Path, trial_metadata: TrialMetadata) -> str:
    return (f'Average of frames {FRAMES_AVG_START}:'
            f'{FRAMES_AVG_END} ({tif_path.name})\nOdors = '
            f'{trial_metadata.odor1_flow}% {trial_metadata.odor1.value}, '
            f'{trial_metadata.odor2_flow}% {trial_metadata.odor2.value}\n'
            f'Target = {trial_metadata.target.value}, '
            f'Response = {trial_metadata.response.value}')


def load_image(tif_path: Path) -> NDArray[np.int16]:
    with tifffile.TiffFile(tif_path) as tf:
        tiff_arr = tf.asarray(range(len(tf.pages)))
    return tiff_arr


def render_frame(tiff_arr: NDArray[np.int16], rois: OrderedDict[ROIName, ROI],
                 rois_focused: list[ROIName], frame: int,
                 smoothen: bool) -> Image:
    if smoothen:
        tiff_image = Image.fromarray(
            np.mean(tiff_arr[frame:frame + SMOOTHING_DISTANCE], axis=0))
    else:
        tiff_image = Image.fromarray(tiff_arr[frame])
    tiff_image_rgb = Image.new('RGB', tiff_image.size)
    tiff_image_rgb.paste(tiff_image)
    draw = ImageDraw.Draw(tiff_image_rgb)
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
    x0, y0 = tiff_image.width - 1, 0
    draw.rectangle([x0 - 100, y0, x0, y0 + 30], fill=(0, 0, 0, 10))
    draw.text([x0 - 5, y0 + 5],
              f'{frame+1}/{tiff_arr.shape[0]}',
              fill='red',
              font_size=24,
              anchor='rt')
    return tiff_image_rgb


def render_thumbnail(tiff_arr: NDArray[np.int16], rois: OrderedDict[ROIName,
                                                                    ROI],
                     rois_focused: list[ROIName]) -> Image:
    tiff_arr_avg = np.mean(tiff_arr[FRAMES_AVG_START:FRAMES_AVG_END], axis=0)
    tiff_image = Image.fromarray(tiff_arr_avg)
    tiff_image_rgb = Image.new('RGB', tiff_image.size)
    tiff_image_rgb.paste(tiff_image)
    draw = ImageDraw.Draw(tiff_image_rgb)
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
    return tiff_image_rgb


def is_in_roi(x: int, y: int, roi: ROI) -> bool:
    half_w, half_h = roi['width'] / 2, roi['height'] / 2
    x_cent, y_cent = roi['left'] + half_w, roi['top'] + half_h
    discrim = ((x - x_cent) / half_w)**2 + ((y - y_cent) / half_h)**2
    return (discrim < 1)


def render_none_plot(fig: Figure):
    fig.clf()


cache_line_behavioral_plot = None


def render_behavioral_plot(fig: Figure, metadata: TrialMetadata,
                           frame: int | None) -> None:
    if frame is None:
        fig.clf()
        ax = fig.add_subplot()
        offset = metadata.frame_times[0]
        ax.plot(metadata.sniff_times - offset,
                metadata.sniff_values,
                label='Sniff')
        ax.axvline(metadata.odor_time - offset, color='black', label='Odor')
        ax.axvspan(0,
                   metadata.frame_times[-1] - offset,
                   alpha=0.1,
                   color='red',
                   label='Video frames')
        m = np.max(metadata.sniff_values)
        ax.vlines(metadata.lick1_times - offset,
                  0,
                  m,
                  color='chocolate',
                  label='Lick left')
        ax.vlines(metadata.lick2_times - offset,
                  -m,
                  0,
                  color='brown',
                  label='Lick right')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Sniff')
        ax.set_title('Behavioral response')
        ax.legend()
    else:
        global cache_line_behavioral_plot
        if cache_line_behavioral_plot is not None:
            cache_line_behavioral_plot.remove()
        offset = metadata.frame_times[0]
        cache_line_behavioral_plot = fig.axes[0].axvline(
            metadata.frame_times[frame] - offset, color='red')


cache_line_fluorescence_plot = None


def render_fluorescence_plot(fig: Figure, metadata: TrialMetadata,
                             tiff_arr: NDArray[np.int16],
                             rois: OrderedDict[ROIName,
                                               ROI], rois_focused: list[ROI],
                             frame: int | None) -> None:
    if frame is None:
        fig.clf()
        if rois_focused:
            ys, xs = np.indices(tiff_arr.shape[1:])
            ax = fig.add_subplot()
            offset = metadata.frame_times[0]
            for i, roi_name in enumerate(rois_focused):
                indices_in_roi = is_in_roi(xs, ys, rois[roi_name])
                f_vals = np.mean(tiff_arr[:, indices_in_roi], axis=1)
                median_f_val = np.median(
                    f_vals[BASELINE_AVG_START:BASELINE_AVG_END])
                df_vals = (f_vals - median_f_val) / median_f_val
                ax.plot(metadata.frame_times - offset,
                        df_vals,
                        label=f'ROI {i+1}')
            ax.axvline(metadata.odor_time - offset,
                       color='black',
                       label='Odor')
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('ΔF/F')
            ax.set_ylim((-1.5, 1.5))
            ax.set_title('Fluoresence of highlighted ROI(s)')
            ax.legend()
        else:
            fig.text(0.5,
                     0.5,
                     'Select an ROI...',
                     fontsize=30,
                     horizontalalignment='center',
                     verticalalignment='center')
    else:
        if fig.axes:
            global cache_line_fluorescence_plot
            if cache_line_fluorescence_plot is not None:
                cache_line_fluorescence_plot.remove()
            offset = metadata.frame_times[0]
            cache_line_fluorescence_plot = fig.axes[0].axvline(
                metadata.frame_times[frame] - offset, color='red')


def render_plot(fig: Figure, plot_setting: PlotSetting,
                metadata: TrialMetadata, tif_arr: NDArray[np.int16],
                rois: OrderedDict[ROIName, ROI], rois_focused: list[ROI],
                frame: int | None) -> None:
    if plot_setting == PlotSetting.NONE:
        render_none_plot(fig)
    if plot_setting == PlotSetting.BEHAVIOR:
        render_behavioral_plot(fig, metadata, frame)
    elif plot_setting == PlotSetting.FLUORESCENCE:
        render_fluorescence_plot(fig, metadata, tif_arr, rois, rois_focused,
                                 frame)


def run_gui() -> None:
    print('Loading H5 file...')
    try:
        with H5_PICKLE_PATH.open('rb') as f:
            h5_data = pickle.load(f)
    except FileNotFoundError:
        h5_data = load_h5()
        with H5_PICKLE_PATH.open('wb') as f:
            pickle.dump(h5_data, f)
    print('H5 file loaded. Preparing GUI...')

    tif_files = sorted([f for f in TIF_FOLDER.iterdir() if f.suffix == '.tif'])
    _tif_file_i = 0
    tif_path = tif_files[_tif_file_i]
    tiff_arr = load_image(tif_path)
    metadata = h5_data[_tif_file_i]
    tk_image_job = None
    resize_job = None
    rois = load_rois()
    rois_focused = []
    plot_setting = PlotSetting.BEHAVIOR
    image = None  # prevent garbage collection of image

    root = tk.Tk()
    root.title('Calcium imaging analyzer')
    root.configure(background='light grey')
    root.geometry(DEFAULT_WINDOW_SIZE)

    style = ttk.Style()
    style.configure('TFrame', background='light blue')
    style.configure('Invisible.TFrame', background='light grey')
    style.configure('TLabel', background='light blue')

    main_frame = ttk.Frame(root, style='Invisible.TFrame')
    main_frame.pack(padx=LARGE_PAD, side='left', fill='y', expand=True)

    ci_frame = ttk.Frame(main_frame)
    ci_frame.pack(pady=LARGE_PAD, fill='both', expand=True)
    ci_heading = ttk.Label(ci_frame,
                           justify='center',
                           text='Calcium image',
                           font=HEADING_FONT)
    ci_heading.pack(pady=(SMALL_PAD, 0))
    ci_label = ttk.Label(ci_frame, justify='center')
    ci_label.pack()
    left_button = ttk.Button(ci_frame, text='<<')
    left_button.pack(side='left', padx=SMALL_PAD)
    right_button = ttk.Button(ci_frame, text='>>')
    right_button.pack(side='right', padx=SMALL_PAD)
    tk_image = tk.Canvas(ci_frame)
    tk_image.pack(pady=SMALL_PAD)

    button_frame = ttk.Frame(ci_frame)
    button_frame.pack(pady=(0, SMALL_PAD))
    play_button = ttk.Button(button_frame, text='Play')
    play_button.pack(side='left', padx=(0, SMALL_PAD))
    play_slow_button = ttk.Button(button_frame, text='Play (slow)')
    play_slow_button.pack(side='left', padx=(0, SMALL_PAD))
    stop_button = ttk.Button(button_frame, text='Stop')
    stop_button.pack(side='left', padx=(0, SMALL_PAD))
    select_all_button = ttk.Button(button_frame, text='Select All')
    select_all_button.pack(side='left', padx=(0, SMALL_PAD))
    jump_to_button = ttk.Button(button_frame, text='Jump to...')
    jump_to_button.pack(side='left', padx=(0, SMALL_PAD))
    jump_to_var = tk.StringVar()
    jump_to_text_box = ttk.Entry(button_frame,
                                 textvariable=jump_to_var,
                                 width=5)
    jump_to_text_box.pack(side='left')

    button2_frame = ttk.Frame(ci_frame)
    button2_frame.pack(pady=(0, SMALL_PAD))
    start_from_odor_var = tk.BooleanVar()
    start_from_odor_button = ttk.Checkbutton(
        button2_frame,
        text='Play from odor presentation?',
        variable=start_from_odor_var)
    start_from_odor_button.pack(side='left', padx=(0, SMALL_PAD))
    smoothen_video_var = tk.BooleanVar()
    smoothen_video_button = ttk.Checkbutton(button2_frame,
                                            text='Smoothen video?',
                                            variable=smoothen_video_var)
    smoothen_video_button.pack(side='left')

    side_frame = ttk.Frame(root, style='Invisible.TFrame')
    side_frame.pack(padx=(0, LARGE_PAD), side='right', fill='y', expand=True)

    display_frame = ttk.Frame(side_frame)
    display_frame.pack(pady=LARGE_PAD, fill='both', expand=True)
    display_heading = ttk.Label(display_frame,
                                text='Analysis',
                                font=HEADING_FONT)
    display_heading.pack(pady=SMALL_PAD)
    fig = Figure()
    canvas = FigureCanvasTkAgg(fig, master=display_frame)
    canvas.get_tk_widget().pack()
    toolbar = NavigationToolbar2Tk(canvas, display_frame, pack_toolbar=False)
    toolbar.pack(pady=(0, MEDIUM_PAD), fill='x')
    plot_button_frame = ttk.Frame(display_frame)
    plot_button_frame.pack()
    plot_var = tk.StringVar(display_frame, plot_setting.value)
    plot_buttons = []
    for i, s in enumerate(PlotSetting):
        button = ttk.Radiobutton(plot_button_frame,
                                 text=s.value,
                                 variable=plot_var,
                                 value=s.value)
        button.pack(padx=int(SMALL_PAD / 2), side='left')
        plot_buttons.append(button)

    def update_job(job: str) -> None:
        nonlocal tk_image_job
        tk_image_job = job

    def stop_job() -> None:
        global cache_line_behavioral_plot, cache_line_fluorescence_plot
        nonlocal tk_image_job
        if tk_image_job:
            tk_image.after_cancel(tk_image_job)
            tk_image_job = None
        if cache_line_behavioral_plot:
            cache_line_behavioral_plot.remove()
            cache_line_behavioral_plot = None
        if cache_line_fluorescence_plot:
            cache_line_fluorescence_plot.remove()
            cache_line_fluorescence_plot = None

    def update_resize_job(job: str) -> None:
        nonlocal resize_job
        resize_job = job

    def stop_resize_job() -> None:
        nonlocal resize_job
        if resize_job:
            root.after_cancel(resize_job)
            resize_job = None

    def get_tif_file_i() -> int:
        return _tif_file_i

    def set_tif_file_i(i: int) -> None:
        nonlocal _tif_file_i, tif_path, tiff_arr, metadata
        _tif_file_i = i % len(tif_files)
        tif_path = tif_files[_tif_file_i]
        tiff_arr = load_image(tif_path)
        metadata = h5_data[_tif_file_i]
        stop_job()

    def set_rois_focused(r: list[ROIName]) -> None:
        nonlocal rois_focused
        rois_focused = r

    def set_plot_setting(s: PlotSetting) -> None:
        nonlocal plot_setting
        plot_setting = s

    def set_ci_image(img: Image) -> None:
        nonlocal image
        size = (tk_image.winfo_reqwidth(), tk_image.winfo_reqheight())
        img_r = img.resize(size)
        image = ImageTk.PhotoImage(img_r)
        tk_image.delete('IMG')
        tk_image.create_image(0, 0, image=image, anchor='nw', tags='IMG')

    def update(redraw_image: bool = True,
               redraw_figure: bool = True,
               frame: int | None = None,
               smoothen: bool = False) -> None:
        if redraw_image:
            ci_label.configure(text=load_caption(tif_path, metadata))
            if frame is None:
                set_ci_image(render_thumbnail(tiff_arr, rois, rois_focused))
            else:
                set_ci_image(
                    render_frame(tiff_arr, rois, rois_focused, frame,
                                 smoothen))
        if redraw_figure:
            render_plot(fig, plot_setting, metadata, tiff_arr, rois,
                        rois_focused, frame)
            canvas.draw()
            toolbar.update()

    def on_left_button() -> None:
        set_tif_file_i(get_tif_file_i() - 1)
        update()

    def on_right_button() -> None:
        set_tif_file_i(get_tif_file_i() + 1)
        update()

    def on_play_button(speed: float) -> None:

        def render(i: int, smoothen: bool) -> None:
            if i + 1 < tiff_arr.shape[0]:
                update(frame=i, smoothen=smoothen)
                dt = int(
                    (metadata.frame_times[i + 1] - metadata.frame_times[i]) /
                    speed)
                update_job(
                    tk_image.after(dt if dt > 0 else 0,
                                   lambda: render(i + 1, smoothen)))
            else:
                stop_job()
                update()

        stop_job()

        if start_from_odor_var.get():
            i = np.argmax(metadata.frame_times > metadata.odor_time)
            render(i - 5 if i > 5 else 0, smoothen_video_var.get())
        else:
            render(0, smoothen_video_var.get())

    def on_stop_button() -> None:
        stop_job()
        update()

    def on_select_all_button() -> None:
        stop_job()
        set_rois_focused(list(rois.keys()))
        update()

    def on_jump_to_button() -> None:
        try:
            i = jump_to_var.get()
            set_tif_file_i(int(i) - 1)
            update()
        except ValueError:
            pass

    def on_roi_click(roi_name: ROIName | None, shift_pressed: bool) -> None:
        stop_job()
        if shift_pressed:
            if roi_name:
                if roi_name in rois_focused:
                    rois_focused.remove(roi_name)
                else:
                    rois_focused.append(roi_name)
        else:
            if roi_name:
                set_rois_focused([roi_name])
            else:
                set_rois_focused([])
        update()

    def on_image_click(event: tk.Event) -> None:
        shift_pressed = (event.state == 1)
        roi_name = None
        x, y = event.x, event.y
        for roi_n, roi in rois.items():
            if is_in_roi(x, y, roi):
                roi_name = roi_n
        on_roi_click(roi_name, shift_pressed)

    def on_plot_setting_button() -> None:
        stop_job()
        set_plot_setting(PlotSetting(plot_var.get()))
        update()

    def on_resize_window(event: tk.Event) -> None:
        if event.widget == root:
            new_width = event.width
            new_height = event.height
            ci_image_size = int(0.6 * min(new_width, new_height))
            tk_image.configure(width=ci_image_size, height=ci_image_size)
            plot_height = int(0.6 * new_height)
            plot_width = int(1.4 * plot_height)
            canvas.get_tk_widget().configure(width=plot_width,
                                             height=plot_height)
            stop_resize_job()
            update_resize_job(root.after(1000, update))

    left_button.configure(command=on_left_button)
    right_button.configure(command=on_right_button)
    play_button.configure(command=lambda: on_play_button(1))
    play_slow_button.configure(command=lambda: on_play_button(0.2))
    stop_button.configure(command=on_stop_button)
    select_all_button.configure(command=on_select_all_button)
    jump_to_button.configure(command=on_jump_to_button)
    tk_image.bind('<Button>', on_image_click)
    for b in plot_buttons:
        b.configure(command=on_plot_setting_button)
    root.bind('<Configure>', on_resize_window)
    update()

    root.mainloop()


if __name__ == '__main__':
    run_gui()
