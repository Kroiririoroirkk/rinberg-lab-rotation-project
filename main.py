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
from tkinter import ttk

import numpy as np
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)
from matplotlib.figure import Figure
from numpy.typing import NDArray
from PIL import Image, ImageTk

import calcium_image
from config import (BASELINE_AVG_END, BASELINE_AVG_START, DEFAULT_SPATIAL_BLUR,
                    DEFAULT_TEMPORAL_BLUR, DEFAULT_WINDOW_SIZE, H5_PATH,
                    H5_PICKLE_PATH, HEADING_FONT, JOB_SCHEDULE_DELAY,
                    LARGE_PAD, MEDIUM_PAD, ROI_ZIP_PATH, SLOW_SPEED, SMALL_PAD,
                    TIF_FOLDER)
from datatypes import PlotSetting, ROIManager, ROIName, TrialMetadata
import plots


def run_gui() -> None:
    print('Loading H5 file...')
    try:
        with H5_PICKLE_PATH.open('rb') as f:
            h5_data = pickle.load(f)
    except FileNotFoundError:
        h5_data = TrialMetadata.list_from_h5(H5_PATH)
        with H5_PICKLE_PATH.open('wb') as f:
            pickle.dump(h5_data, f)
    print('H5 file loaded. Preparing GUI...')

    tif_files = sorted([f for f in TIF_FOLDER.iterdir() if f.suffix == '.tif'])
    _tif_file_i = 0
    tif_path = tif_files[_tif_file_i]
    tiff_arr = calcium_image.load_image(tif_path)
    _median_tiff_arr = None
    metadata = h5_data[_tif_file_i]
    tk_image_job = None
    _update_figure = False
    _update_job = None
    rois = ROIManager.from_zip(ROI_ZIP_PATH)
    rois_focused = []
    plot_setting = PlotSetting.NONE
    image = None  # prevent garbage collection of image
    _window_width = 0
    _window_height = 0

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
    start_from_odor_var = tk.BooleanVar(value=True)
    start_from_odor_button = ttk.Checkbutton(
        button2_frame,
        text='Play from odor presentation?',
        variable=start_from_odor_var)
    start_from_odor_button.pack(side='left', padx=(0, SMALL_PAD))
    plot_delta_var = tk.BooleanVar(value=True)
    plot_delta_button = ttk.Checkbutton(button2_frame,
                                        text='Plot ΔF/F?',
                                        variable=plot_delta_var)
    plot_delta_button.pack(side='left', padx=(0, SMALL_PAD))
    hide_rois_var = tk.BooleanVar()
    hide_rois_button = ttk.Checkbutton(button2_frame,
                                       text='Hide ROIs?',
                                       variable=hide_rois_var)
    hide_rois_button.pack(side='left')

    button3_frame = ttk.Frame(ci_frame)
    button3_frame.pack()
    spatial_blur_label = ttk.Label(button3_frame, text='Spatial blur:')
    spatial_blur_label.pack(side='left', padx=(0, SMALL_PAD))
    spatial_blur_var = tk.DoubleVar(value=DEFAULT_SPATIAL_BLUR)
    spatial_blur_slider = tk.Scale(button3_frame,
                                   from_=0,
                                   to=3,
                                   resolution=0.1,
                                   orient='horizontal',
                                   variable=spatial_blur_var)
    spatial_blur_slider.pack(side='left', padx=(0, SMALL_PAD))
    temporal_blur_label = ttk.Label(button3_frame, text='Temporal blur:')
    temporal_blur_label.pack(side='left', padx=(0, SMALL_PAD))
    temporal_blur_var = tk.IntVar(value=DEFAULT_TEMPORAL_BLUR)
    temporal_blur_slider = tk.Scale(button3_frame,
                                    from_=0,
                                    to=10,
                                    orient='horizontal',
                                    variable=temporal_blur_var)
    temporal_blur_slider.pack(side='left', padx=(0, SMALL_PAD))

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

    def set_image_job(job: str) -> None:
        nonlocal tk_image_job
        tk_image_job = job

    def stop_image_job() -> None:
        nonlocal tk_image_job
        if tk_image_job:
            tk_image.after_cancel(tk_image_job)
            tk_image_job = None
        plots.delete_running_lines()

    def cue_update_job(update_figure: bool = True) -> None:
        nonlocal _update_figure, _update_job

        def _update():
            nonlocal _update_figure, _update_job
            update(redraw_figure=_update_figure)
            _update_job = None
            _update_figure = False

        if _update_job:
            root.after_cancel(_update_job)
        _update_figure = _update_figure or update_figure
        _update_job = root.after(JOB_SCHEDULE_DELAY, _update)

    def get_tif_file_i() -> int:
        return _tif_file_i

    def set_tif_file_i(i: int) -> None:
        nonlocal _tif_file_i, tif_path, tiff_arr, metadata
        _tif_file_i = i % len(tif_files)
        tif_path = tif_files[_tif_file_i]
        tiff_arr = calcium_image.load_image(tif_path)
        metadata = h5_data[_tif_file_i]
        stop_image_job()

    def get_median_tiff_arr() -> NDArray[np.int16]:
        nonlocal _median_tiff_arr
        if _median_tiff_arr is None:
            _median_tiff_arr = np.median(
                tiff_arr[BASELINE_AVG_START:BASELINE_AVG_END], axis=0)
        return _median_tiff_arr

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

    def was_window_resized(width: int, height: int) -> bool:
        return _window_width != width or _window_height != height

    def set_window_size(width: int, height: int) -> None:
        nonlocal _window_width, _window_height
        _window_width = width
        _window_height = height

    def update(redraw_image: bool = True,
               redraw_figure: bool = True,
               frame: int | None = None) -> None:
        if redraw_image:
            ci_label.configure(
                text=calcium_image.make_caption(tif_path, metadata))
            hide_rois = hide_rois_var.get()
            spatial_blur = spatial_blur_var.get()
            temporal_blur = temporal_blur_var.get()
            median_tiff_arr = get_median_tiff_arr() if plot_delta_var.get(
            ) else None
            if frame is None:
                set_ci_image(
                    calcium_image.render_thumbnail(tiff_arr, metadata, rois,
                                                   rois_focused, hide_rois,
                                                   spatial_blur,
                                                   median_tiff_arr))
            else:
                set_ci_image(
                    calcium_image.render_frame(tiff_arr, rois, rois_focused,
                                               hide_rois, frame, spatial_blur,
                                               temporal_blur, median_tiff_arr))
        if redraw_figure:
            plots.render_plot(fig, plot_setting, metadata, tiff_arr, rois,
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

        def render(i: int) -> None:
            if i + 1 < tiff_arr.shape[0]:
                update(frame=i)
                dt = int(
                    (metadata.frame_times[i + 1] - metadata.frame_times[i]) /
                    speed)
                set_image_job(
                    tk_image.after(dt if dt > 0 else 0, lambda: render(i + 1)))
            else:
                stop_image_job()
                update()

        stop_image_job()

        if start_from_odor_var.get():
            i = np.argmax(metadata.frame_times > metadata.odor_time)
            render(i - 5 if i > 5 else 0)
        else:
            render(0)

    def on_stop_button() -> None:
        stop_image_job()
        update()

    def on_select_all_button() -> None:
        stop_image_job()
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
        stop_image_job()
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
            if ROIManager.is_in_roi(x, y, roi):
                roi_name = roi_n
        on_roi_click(roi_name, shift_pressed)

    def on_plot_setting_button() -> None:
        stop_image_job()
        set_plot_setting(PlotSetting(plot_var.get()))
        update()

    def on_resize_window(event: tk.Event) -> None:
        if event.widget == root:
            new_width = event.width
            new_height = event.height
            if was_window_resized(new_width, new_height):
                set_window_size(new_width, new_height)
                ci_image_size = int(0.6 * min(new_width, new_height))
                tk_image.configure(width=ci_image_size, height=ci_image_size)
                plot_height = int(0.6 * new_height)
                plot_width = int(1.4 * plot_height)
                canvas.get_tk_widget().configure(width=plot_width,
                                                 height=plot_height)
                cue_update_job()

    left_button.configure(command=on_left_button)
    right_button.configure(command=on_right_button)
    play_button.configure(command=lambda: on_play_button(1))
    play_slow_button.configure(command=lambda: on_play_button(SLOW_SPEED))
    stop_button.configure(command=on_stop_button)
    select_all_button.configure(command=on_select_all_button)
    jump_to_button.configure(command=on_jump_to_button)
    plot_delta_button.configure(
        command=lambda: cue_update_job(update_figure=False))
    hide_rois_button.configure(
        command=lambda: cue_update_job(update_figure=False))
    spatial_blur_slider.configure(
        command=lambda _: cue_update_job(update_figure=False))
    temporal_blur_slider.configure(
        command=lambda _: cue_update_job(update_figure=False))
    tk_image.bind('<Button>', on_image_click)
    for b in plot_buttons:
        b.configure(command=on_plot_setting_button)
    root.bind('<Configure>', on_resize_window)
    update()

    root.mainloop()


if __name__ == '__main__':
    run_gui()
