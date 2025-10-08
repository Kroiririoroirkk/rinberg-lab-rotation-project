"""Calcium Imaging Analyzer

The purpose of this script is to allow for the convenient viewing and analysis
of two-photon calcium imaging data, together with behavioral (two-alternative
forced choice task responses) and physiological data (respiration patterns).

Author: Eric Tao (Eric.Tao@nyulangone.org)
Date created: 2025-09-23
Date last updated: 2025-10-08
"""
import pickle
import tkinter as tk
from tkinter import ttk

import numpy as np
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)
from matplotlib.figure import Figure
from PIL import Image, ImageTk

import calcium_image
import plots
from config import (DEFAULT_SPATIAL_BLUR, DEFAULT_TEMPORAL_BLUR,
                    DEFAULT_WINDOW_SIZE, H5_PATH, H5_PICKLE_PATH, HEADING_FONT,
                    JOB_SCHEDULE_DELAY, LARGE_PAD, MEDIUM_PAD, ROI_ZIP_PATH,
                    SLOW_SPEED, SMALL_PAD, TIF_FOLDER)
from datatypes import (PlotSetting, ROIManager, ROIName, TrialMetadata,
                       VariableSet)
from gui_state import GUIState


def run_gui() -> None:
    root = tk.Tk()
    root.title('Calcium imaging analyzer')
    root.configure(background='light grey')
    root.geometry(DEFAULT_WINDOW_SIZE)
    root_update_job = None
    root_updating_figure = False

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
    tk_image_job = None
    tk_image_image = None

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
    hide_rois_button.pack(side='left', padx=(0, SMALL_PAD))
    save_tif_button = ttk.Button(button2_frame, text='Save as TIF')
    save_tif_button.pack(side='left')

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
    plot_setting_var = tk.StringVar(value=PlotSetting.NONE.value)
    plot_buttons = []
    for i, s in enumerate(PlotSetting):
        button = ttk.Radiobutton(plot_button_frame,
                                 text=s.value,
                                 variable=plot_setting_var,
                                 value=s.value)
        button.pack(padx=int(SMALL_PAD / 2), side='left')
        plot_buttons.append(button)

    print('Loading H5 file...')
    try:
        with H5_PICKLE_PATH.open('rb') as f:
            h5_data = pickle.load(f)
    except FileNotFoundError:
        h5_data = TrialMetadata.list_from_h5(H5_PATH)
        with H5_PICKLE_PATH.open('wb') as f:
            pickle.dump(h5_data, f)
    tif_files = sorted([f for f in TIF_FOLDER.iterdir() if f.suffix == '.tif'])
    variable_set = VariableSet(hide_rois_var=hide_rois_var,
                               plot_delta_var=plot_delta_var,
                               spatial_blur_var=spatial_blur_var,
                               temporal_blur_var=temporal_blur_var,
                               plot_setting_var=plot_setting_var)
    gui_state = GUIState(tif_files, h5_data, ROI_ZIP_PATH, variable_set)

    def set_ci_image(img: Image) -> None:
        nonlocal tk_image_image
        size = (tk_image.winfo_reqwidth(), tk_image.winfo_reqheight())
        img_r = img.resize(size)
        tk_image_image = ImageTk.PhotoImage(img_r)
        tk_image.delete('IMG')
        tk_image.create_image(0,
                              0,
                              image=tk_image_image,
                              anchor='nw',
                              tags='IMG')

    def update(redraw_image: bool = True,
               redraw_figure: bool = True,
               frame: int | None = None) -> None:
        message = gui_state.make_message()
        if redraw_image:
            ci_label.configure(text=calcium_image.make_caption(message))
            set_ci_image(calcium_image.render_ci(message, frame))
        if redraw_figure:
            plots.render_plot(fig, message, frame)
            canvas.draw()
            toolbar.update()

    def set_image_job(job: str | None) -> None:
        nonlocal tk_image_job
        tk_image_job = job

    def stop_image_job() -> None:
        if tk_image_job is not None:
            tk_image.after_cancel(tk_image_job)
            set_image_job(None)
        plots.delete_running_lines()

    def cue_update_job(update_figure: bool = True) -> None:
        nonlocal root_update_job, root_updating_figure

        def _update():
            nonlocal root_update_job, root_updating_figure
            update(redraw_figure=update_figure)
            root_update_job = None
            root_updating_figure = False

        if root_update_job is not None:
            root.after_cancel(root_update_job)
        root_updating_figure = root_updating_figure or update_figure
        root_update_job = root.after(JOB_SCHEDULE_DELAY, _update)

    def on_left_button() -> None:
        gui_state.set_tif_file_i(gui_state.get_tif_file_i() - 1)
        stop_image_job()
        update()

    def on_right_button() -> None:
        gui_state.set_tif_file_i(gui_state.get_tif_file_i() + 1)
        stop_image_job()
        update()

    def on_play_button(speed: float) -> None:

        def render(i: int) -> None:
            if i + 1 < gui_state.get_tiff_arr().shape[0]:
                update(frame=i)
                dt = int((gui_state.get_metadata().frame_times[i + 1] -
                          gui_state.get_metadata().frame_times[i]) / speed)
                set_image_job(
                    tk_image.after(dt if dt > 0 else 0, lambda: render(i + 1)))
            else:
                stop_image_job()
                update()

        stop_image_job()

        if start_from_odor_var.get():
            i = np.argmax(gui_state.get_metadata().frame_times >
                          gui_state.get_metadata().odor_time)
            render(i - 5 if i > 5 else 0)
        else:
            render(0)

    def on_stop_button() -> None:
        stop_image_job()
        update()

    def on_select_all_button() -> None:
        stop_image_job()
        gui_state.set_rois_focused(list(gui_state.get_all_rois().keys()))
        update()

    def on_jump_to_button() -> None:
        try:
            i = jump_to_var.get()
            gui_state.set_tif_file_i(int(i) - 1)
            stop_image_job()
            update()
        except ValueError:
            pass

    def on_save_tif_button() -> None:
        calcium_image.export_tif(tif_files, h5_data, gui_state.make_message())

    def on_roi_click(roi_name: ROIName | None, shift_pressed: bool) -> None:
        stop_image_job()
        if shift_pressed:
            if roi_name:
                gui_state.toggle_roi(roi_name)
        else:
            if roi_name:
                gui_state.set_rois_focused([roi_name])
            else:
                gui_state.set_rois_focused([])
        update()

    def on_image_click(event: tk.Event) -> None:
        shift_pressed = (event.state == 1)
        roi_name = None
        x, y = event.x, event.y
        for roi_n, roi in gui_state.get_all_rois().items():
            if ROIManager.is_in_roi(x, y, roi):
                roi_name = roi_n
        on_roi_click(roi_name, shift_pressed)

    def on_plot_setting_button() -> None:
        stop_image_job()
        update()

    def on_resize_window(event: tk.Event) -> None:
        if event.widget == root:
            new_width = event.width
            new_height = event.height
            if gui_state.was_window_resized(new_width, new_height):
                gui_state.set_window_size(new_width, new_height)
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
    save_tif_button.configure(command=on_save_tif_button)
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
