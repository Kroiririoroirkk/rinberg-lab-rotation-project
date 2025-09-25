"""Calcium Imaging Analyzer

The purpose of this script is to allow for the convenient viewing and analysis
of two-photon calcium imaging data, together with behavioral (two-alternative
forced choice task responses) and physiological data (respiration patterns).

Author: Eric Tao (Eric.Tao@nyulangone.org)
Date created: 2025-09-23
Date last updated: 2025-09-23
"""

import pathlib
import tkinter as tk
from dataclasses import dataclass
from tkinter import ttk

import h5py
import numpy as np
import read_roi
import tifffile
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)
from matplotlib.figure import Figure
from PIL import Image, ImageDraw, ImageTk

ROI_ZIP_PATH = pathlib.Path('data/GLOM/ROIs/RoiSet.zip')
TIF_FOLDER = pathlib.Path('data/GLOM/TIFFs')
H5_PATH = pathlib.Path('data/GLOM/mouse0953_sess20_D2025_8_18T12_36_43.h5')
FRAMES_AVG_START = 0
FRAMES_AVG_END = 100
BASELINE_AVG_START = 0
BASELINE_AVG_END = 100
DEFAULT_WINDOW_SIZE = '1600x800'


@dataclass
class TrialMetadata:
    lick1_times: list[int]  # in ms
    lick2_times: list[int]  # in ms
    sniff_cycle: list[int]


def load_h5():
    trial_metadata_list = []
    f = h5py.File(H5_PATH, 'r')
    recorded_trials = np.nonzero(f['Trials']['record'])[0]
    for i in recorded_trials:
        trial_f = f[f'Trial{i+1:04d}']
        lick1_times = np.concatenate(
            trial_f['lick1']) if trial_f['lick1'].size else []
        lick2_times = np.concatenate(
            trial_f['lick2']) if trial_f['lick2'].size else []
        sniff_cycle = np.concatenate(
            trial_f['sniff']) if trial_f['sniff'].size else []
        trial_metadata_list.append(
            TrialMetadata(lick1_times=lick1_times,
                          lick2_times=lick2_times,
                          sniff_cycle=sniff_cycle))
    return trial_metadata_list


def load_rois():
    rois = read_roi.read_roi_zip(ROI_ZIP_PATH)
    for v in rois.values():
        if v.get('type') != 'oval':
            raise ValueError('Cannot process non-oval ROI.')
    return rois


def load_caption(tif_path):
    return (f'Calcium image (average of frames {FRAMES_AVG_START}:'
            f'{FRAMES_AVG_END})\n{tif_path.name}')


def load_image(tif_path):
    with tifffile.TiffFile(tif_path) as tf:
        tiff_arr = tf.asarray(range(len(tf.pages)))
    return tiff_arr


def render_frame(tiff_arr, rois, rois_focused, frame):
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
    # If we don't use a nonlocal variable here, the image will be
    # garbage-collected
    global tiff_photo_image
    tiff_photo_image = ImageTk.PhotoImage(image=tiff_image_rgb)
    return tiff_photo_image


def render_thumbnail(tif_path, rois, rois_focused):
    tiff_arr = load_image(tif_path)
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
    # If we don't use a nonlocal variable here, the image will be
    # garbage-collected
    global tiff_photo_image
    tiff_photo_image = ImageTk.PhotoImage(image=tiff_image_rgb)
    return tiff_photo_image


def is_in_roi(x, y, roi):
    half_w, half_h = roi['width'] / 2, roi['height'] / 2
    x_cent, y_cent = roi['left'] + half_w, roi['top'] + half_h
    discrim = ((x - x_cent) / half_w)**2 + ((y - y_cent) / half_h)**2
    return (discrim < 1)


def render_fluorescence_plot(fig, tif_path, rois, rois_focused):
    fig.clf()
    if rois_focused:
        tiff_arr = load_image(tif_path)
        xs, ys = np.indices(tiff_arr.shape[1:])
        ax = fig.add_subplot()
        for i, roi_name in enumerate(rois_focused):
            mask_one_frame = np.logical_not(is_in_roi(xs, ys, rois[roi_name]))
            mask = np.tile(mask_one_frame, (tiff_arr.shape[0], 1, 1))
            masked_arr = np.ma.masked_array(tiff_arr, mask)
            f_vals = masked_arr.mean(axis=(1, 2)).data
            median_f_val = np.median(
                f_vals[BASELINE_AVG_START:BASELINE_AVG_END])
            df_vals = (f_vals - median_f_val) / median_f_val
            ax.plot(df_vals, label=f'ROI {i+1}')
        ax.set_xlabel('Frame')
        ax.set_ylabel('ΔF/F')
        ax.set_ylim((-1, 1))
        ax.set_title('Fluoresence of highlighted ROI(s)')
        ax.legend()


def run_gui():
    rois = load_rois()
    rois_focused = []
    tif_files = sorted([f for f in TIF_FOLDER.iterdir() if f.suffix == '.tif'])
    tif_file_i = 0

    print('Loading H5 file...')
    h5_data = load_h5()

    root = tk.Tk()
    root.title('Calcium imaging analyzer')
    root.configure(background='light grey')
    root.geometry(DEFAULT_WINDOW_SIZE)

    style = ttk.Style()
    style.configure('TFrame', background='light blue')
    style.configure('Invisible.TFrame', background='light grey')
    style.configure('TLabel', background='light blue')

    main_frame = ttk.Frame(root, style='Invisible.TFrame')
    main_frame.pack(padx=50, side='left', fill='y')

    ci_frame = ttk.Frame(main_frame)
    ci_frame.pack(pady=50, fill='x')
    ci_label = ttk.Label(ci_frame, justify='center')
    ci_label.pack(pady=(10, 0))
    left_button = ttk.Button(ci_frame, text='<<')
    left_button.pack(side='left', padx=10)
    right_button = ttk.Button(ci_frame, text='>>')
    right_button.pack(side='right', padx=10)
    tk_image = ttk.Label(ci_frame)
    tk_image.pack(pady=10)
    tk_image_job = None
    button_frame = ttk.Frame(ci_frame)
    button_frame.pack(pady=(0, 10))
    play_button = ttk.Button(button_frame, text='Play')
    play_button.pack(side='left', padx=(0, 10))
    stop_button = ttk.Button(button_frame, text='Stop')
    stop_button.pack(side='right')

    behavior_frame = ttk.Frame(main_frame)
    behavior_frame.pack(fill='x')
    behavior_label = ttk.Label(behavior_frame, text='Behavior')
    behavior_label.pack(pady=10)

    side_frame = ttk.Frame(root, style='Invisible.TFrame')
    side_frame.pack(padx=(0, 50), side='right', fill='y')

    display_frame = ttk.Frame(side_frame)
    display_frame.pack(pady=50, fill='both', expand=True)
    display_label = ttk.Label(display_frame, text='Display')
    display_label.pack(pady=10)
    fig = Figure()
    canvas = FigureCanvasTkAgg(fig, master=display_frame)
    toolbar = NavigationToolbar2Tk(canvas, display_frame)

    def update():
        tif_path = tif_files[tif_file_i]
        ci_label.configure(text=load_caption(tif_path))
        tk_image.configure(
            image=render_thumbnail(tif_path, rois, rois_focused))
        render_fluorescence_plot(fig, tif_path, rois, rois_focused)
        canvas.draw()
        toolbar.update()
        canvas.get_tk_widget().pack()

    def on_left_button():
        nonlocal tif_file_i
        tif_file_i = tif_file_i - 1
        update()

    def on_right_button():
        nonlocal tif_file_i
        tif_file_i = tif_file_i + 1
        update()

    def on_play_button():
        nonlocal tk_image_job
        tiff_arr = load_image(tif_files[tif_file_i])

        def render(i):
            tk_image.configure(
                image=render_frame(tiff_arr, rois, rois_focused, i))
            if i + 1 < tiff_arr.shape[0]:
                nonlocal tk_image_job
                tk_image_job = tk_image.after(
                    10, lambda: render(i + 1))  # 10 ms is arbitrary
            else:
                update()

        if tk_image_job:
            tk_image.after_cancel(tk_image_job)
            tk_image_job = None

        render(0)

    def on_stop_button():
        nonlocal tk_image_job
        if tk_image_job:
            tk_image.after_cancel(tk_image_job)
            tk_image_job = None
        update()

    def on_roi_click(roi_name, shift_pressed):
        nonlocal rois_focused
        if shift_pressed:
            if roi_name:
                if roi_name in rois_focused:
                    rois_focused.remove(roi_name)
                else:
                    rois_focused.append(roi_name)
        else:
            if roi_name:
                rois_focused = [roi_name]
            else:
                rois_focused = []
        update()

    def on_image_click(event):
        shift_pressed = (event.state == 1)
        roi_name = None
        x, y = event.x, event.y
        for roi_n, roi in rois.items():
            if is_in_roi(x, y, roi):
                roi_name = roi_n
        on_roi_click(roi_name, shift_pressed)

    left_button.configure(command=on_left_button)
    right_button.configure(command=on_right_button)
    play_button.configure(command=on_play_button)
    stop_button.configure(command=on_stop_button)
    tk_image.bind('<Button>', on_image_click)
    update()

    root.mainloop()


if __name__ == '__main__':
    run_gui()
