import tkinter as tk
import tkinter.ttk as ttk
from typing import Any, Self

from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)
from matplotlib.figure import Figure
from PIL import Image, ImageTk

from config import (DEFAULT_WINDOW_SIZE, HEADING_FONT, JOB_SCHEDULE_DELAY,
                    LARGE_PAD, SLOW_SPEED, SMALL_PAD)
from datatypes import (ByStimIDPlotSetting, ByTrialPlotSetting, PageSetting,
                       for_page)

PAGE_SETTING_VIEW_DICT: dict[PageSetting, type] = dict()


class FlowFrame(ttk.Frame):

    def __init__(self, master, padding, **kwargs):
        super().__init__(master, **kwargs)
        self.padding = padding
        self.bind("<Configure>", self._on_configure)

    def _on_configure(self, event=None):
        x, y, row_height = self.padding, self.padding, 0
        width = self.winfo_width()

        for widget in self.winfo_children():
            widget.update_idletasks()
            w, h = widget.winfo_reqwidth(), widget.winfo_reqheight()

            # Wrap to next line unless one widget takes up more than
            # the entire width already
            if (x + w + self.padding
                    > width) and (self.padding + w + self.padding < width):
                x = self.padding
                y += row_height + self.padding
                row_height = 0

            widget.place(x=x, y=y, width=w, height=h)
            x += w + self.padding
            row_height = max(row_height, h)

        self.config(height=y + row_height + self.padding)


@for_page(PAGE_SETTING_VIEW_DICT, PageSetting.BY_TRIAL)
class ByTrialView(ttk.Frame):

    def __init__(self: Self, parent: 'View', **kwargs) -> None:
        super().__init__(parent.container, **kwargs)
        self.parent = parent
        self.controller = None
        self.root_update_job = None
        self.root_updating_figure = False
        self.create_widgets()

    def create_widgets(self: Self) -> None:

        self.main_frame = ttk.Frame(self, style='Invisible.TFrame')
        self.main_frame.pack(padx=LARGE_PAD,
                             side='left',
                             fill='y',
                             expand=True)

        self.ci_frame = ttk.Frame(self.main_frame)
        self.ci_frame.pack(pady=LARGE_PAD, fill='both', expand=True)
        self.ci_heading = ttk.Label(self.ci_frame,
                                    justify='center',
                                    text='Calcium image',
                                    font=HEADING_FONT)
        self.ci_heading.pack(pady=(SMALL_PAD, 0))
        self.ci_label = ttk.Label(self.ci_frame, justify='center')
        self.ci_label.pack()
        self.left_button = ttk.Button(self.ci_frame, text='<<')
        self.left_button.pack(side='left', padx=SMALL_PAD)
        self.left_button.configure(command=self.on_left_button)
        self.right_button = ttk.Button(self.ci_frame, text='>>')
        self.right_button.pack(side='right', padx=SMALL_PAD)
        self.right_button.configure(command=self.on_right_button)
        self.tk_image = tk.Canvas(self.ci_frame)
        self.tk_image.pack(pady=SMALL_PAD)
        self.tk_image.bind('<Button>', self.on_image_click)
        self.tk_image_job = None
        self.tk_image_image = None
        self.image_original = None
        self.image_resized = None

        self.button_frame = ttk.Frame(self.ci_frame)
        self.button_frame.pack(pady=(0, SMALL_PAD))
        self.play_button = ttk.Button(self.button_frame, text='Play')
        self.play_button.pack(side='left', padx=(0, SMALL_PAD))
        self.play_button.configure(command=self.on_play_button(1))
        self.play_slow_button = ttk.Button(self.button_frame,
                                           text='Play (slow)')
        self.play_slow_button.pack(side='left', padx=(0, SMALL_PAD))
        self.play_slow_button.configure(
            command=self.on_play_button(SLOW_SPEED))
        self.stop_button = ttk.Button(self.button_frame, text='Stop')
        self.stop_button.pack(side='left', padx=(0, SMALL_PAD))
        self.stop_button.configure(command=self.on_stop_button)
        self.select_all_button = ttk.Button(self.button_frame,
                                            text='Select All')
        self.select_all_button.pack(side='left', padx=(0, SMALL_PAD))
        self.select_all_button.configure(command=self.on_select_all_button)
        self.jump_to_button = ttk.Button(self.button_frame, text='Jump to...')
        self.jump_to_button.pack(side='left', padx=(0, SMALL_PAD))
        self.jump_to_button.configure(command=self.on_jump_to_button)
        self.jump_to_var = tk.StringVar()
        self.jump_to_text_box = ttk.Entry(self.button_frame,
                                          textvariable=self.jump_to_var,
                                          width=5)
        self.jump_to_text_box.pack(side='left')

        self.button2_frame = ttk.Frame(self.ci_frame)
        self.button2_frame.pack(pady=(0, SMALL_PAD))
        self.start_from_odor_var = tk.BooleanVar()
        self.start_from_odor_button = ttk.Checkbutton(
            self.button2_frame,
            text='Play from odor presentation?',
            variable=self.start_from_odor_var)
        self.start_from_odor_button.pack(side='left', padx=(0, SMALL_PAD))
        self.plot_delta_var = tk.BooleanVar()
        self.plot_delta_button = ttk.Checkbutton(self.button2_frame,
                                                 text='Plot ΔF/F?',
                                                 variable=self.plot_delta_var)
        self.plot_delta_button.pack(side='left', padx=(0, SMALL_PAD))
        self.plot_delta_button.configure(command=self.cue_update_job_wrapper(
            update_figure=False))
        self.hide_rois_var = tk.BooleanVar()
        self.hide_rois_button = ttk.Checkbutton(self.button2_frame,
                                                text='Hide ROIs?',
                                                variable=self.hide_rois_var)
        self.hide_rois_button.pack(side='left', padx=(0, SMALL_PAD))
        self.hide_rois_button.configure(command=self.cue_update_job_wrapper(
            update_figure=False))
        self.save_tiff_button = ttk.Button(self.button2_frame,
                                           text='Save as TIFF')
        self.save_tiff_button.pack(side='left')
        self.save_tiff_button.configure(command=self.on_save_tiff_button)

        self.button3_frame = ttk.Frame(self.ci_frame)
        self.button3_frame.pack()
        self.spatial_blur_label = ttk.Label(self.button3_frame,
                                            text='Spatial blur:')
        self.spatial_blur_label.pack(side='left', padx=(0, SMALL_PAD))
        self.spatial_blur_var = tk.DoubleVar()
        self.spatial_blur_slider = tk.Scale(self.button3_frame,
                                            from_=0,
                                            to=3,
                                            resolution=0.1,
                                            orient='horizontal',
                                            variable=self.spatial_blur_var)
        self.spatial_blur_slider.pack(side='left', padx=(0, SMALL_PAD))
        self.spatial_blur_slider.configure(command=self.cue_update_job_wrapper(
            update_figure=False))
        self.temporal_blur_label = ttk.Label(self.button3_frame,
                                             text='Temporal blur:')
        self.temporal_blur_label.pack(side='left', padx=(0, SMALL_PAD))
        self.temporal_blur_var = tk.IntVar()
        self.temporal_blur_slider = tk.Scale(self.button3_frame,
                                             from_=0,
                                             to=10,
                                             orient='horizontal',
                                             variable=self.temporal_blur_var)
        self.temporal_blur_slider.pack(side='left', padx=(0, SMALL_PAD))
        self.temporal_blur_slider.configure(
            command=self.cue_update_job_wrapper(update_figure=False))

        self.side_frame = ttk.Frame(self, style='Invisible.TFrame')
        self.side_frame.pack(padx=(0, LARGE_PAD),
                             pady=(LARGE_PAD, LARGE_PAD),
                             side='right',
                             fill='y',
                             expand=True)

        self.display_frame = ttk.Frame(self.side_frame)
        self.display_frame.pack(fill='x')
        self.display_heading = ttk.Label(self.display_frame,
                                         text='Analysis',
                                         font=HEADING_FONT)
        self.display_heading.pack(pady=SMALL_PAD)
        self.fig = Figure()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.display_frame)
        self.canvas.get_tk_widget().pack()
        self.toolbar = NavigationToolbar2Tk(self.canvas,
                                            self.display_frame,
                                            pack_toolbar=False)
        self.toolbar.pack(pady=(0, SMALL_PAD), fill='x')
        self.plot_button_frame = FlowFrame(self.display_frame, SMALL_PAD)
        self.plot_button_frame.pack(fill='x', expand=True)
        self.plot_setting_var = tk.StringVar()
        self.plot_buttons = []
        for i, s in enumerate(ByTrialPlotSetting):
            button = ttk.Radiobutton(self.plot_button_frame,
                                     text=s.value,
                                     variable=self.plot_setting_var,
                                     value=s.value)
            button.configure(command=self.on_plot_setting_button)
            self.plot_buttons.append(button)

        self.page_switcher_frame = ttk.Frame(self.side_frame)
        self.page_switcher_frame.pack(side='bottom', fill='x')
        self.page_switcher_label = ttk.Label(self.page_switcher_frame,
                                             text='View',
                                             font=HEADING_FONT)
        self.page_switcher_label.pack()
        self.page_button_frame = ttk.Frame(self.page_switcher_frame)
        self.page_button_frame.pack(pady=SMALL_PAD)
        self.page_buttons = []
        for i, s in enumerate(PageSetting):
            button = ttk.Radiobutton(self.page_button_frame,
                                     text=s.value,
                                     variable=self.parent.page_setting_var,
                                     value=s.value)
            button.pack(padx=int(SMALL_PAD / 2), side='left')
            button.configure(command=self.on_page_setting_button)
            self.page_buttons.append(button)

    @property
    def original_image_size(self: Self) -> tuple[int, int]:
        return self.image_original.size

    @property
    def displayed_image_size(self: Self) -> tuple[int, int]:
        return self.image_resized.size

    def set_ci_image(self: Self, caption: str, img: Image.Image) -> None:
        self.ci_label.configure(text=caption)
        self.image_original = img
        new_size = (self.tk_image.winfo_reqwidth(),
                    self.tk_image.winfo_reqheight())
        self.image_resized = img.resize(new_size)
        self.tk_image_image = ImageTk.PhotoImage(self.image_resized)
        self.tk_image.delete('IMG')
        self.tk_image.create_image(0,
                                   0,
                                   image=self.tk_image_image,
                                   anchor='nw',
                                   tags='IMG')

    def update_fig(self: Self) -> None:
        self.canvas.draw()
        self.toolbar.update()

    def stop_image_job(self: Self) -> None:
        if self.tk_image_job is not None:
            self.tk_image.after_cancel(self.tk_image_job)
            self.tk_image_job = None
        self.controller.delete_running_lines()

    def cue_update_job(self: Self, update_figure: bool = True) -> None:

        def _update():
            self.controller.update(redraw_figure=update_figure)
            self.root_update_job = None
            self.root_updating_figure = False

        if self.root_update_job is not None:
            self.after_cancel(self.root_update_job)
        self.root_updating_figure = self.root_updating_figure or update_figure
        self.root_update_job = self.after(JOB_SCHEDULE_DELAY, _update)

    def cue_update_job_wrapper(self: Self, update_figure: bool = True) -> None:

        def wrapper(*args: list[Any]):
            self.cue_update_job(update_figure=update_figure)

        return wrapper

    def on_left_button(self: Self) -> None:
        self.controller.decrement_tiff_file()
        self.stop_image_job()
        self.controller.update()

    def on_right_button(self: Self) -> None:
        self.controller.increment_tiff_file()
        self.stop_image_job()
        self.controller.update()

    def on_play_button(self: Self, speed: float) -> None:

        def wrapper() -> None:
            self.controller.play_ci_video(speed)

        return wrapper

    def on_stop_button(self: Self) -> None:
        self.stop_image_job()
        self.controller.update()

    def on_select_all_button(self: Self) -> None:
        self.stop_image_job()
        self.controller.select_all_rois()
        self.controller.update()

    def on_jump_to_button(self: Self) -> None:
        try:
            i = self.jump_to_var.get()
        except ValueError:
            return
        self.controller.jump_to_file(int(i) - 1)
        self.stop_image_job()
        self.controller.update()

    def on_save_tiff_button(self: Self) -> None:
        self.controller.save_tiff()

    def on_image_click(self: Self, event: tk.Event) -> None:
        shift_pressed = (event.state == 1)
        orig_w, orig_h = self.original_image_size
        display_w, display_h = self.displayed_image_size
        x, y = (event.x / display_w) * orig_w, (event.y / display_h) * orig_h
        self.controller.process_image_click(shift_pressed, x, y)
        self.controller.update()

    def on_plot_setting_button(self: Self) -> None:
        self.stop_image_job()
        self.controller.update()

    def on_page_setting_button(self: Self) -> None:
        self.stop_image_job()
        self.parent.update_frame()
        self.controller.update()

    def on_resize_window(self: Self, width: int, height: int) -> None:
        ci_image_size = int(0.6 * min(width, height))
        self.tk_image.configure(width=ci_image_size, height=ci_image_size)
        plot_height = int(0.6 * height)
        plot_width = int(1.4 * plot_height)
        self.canvas.get_tk_widget().configure(width=plot_width,
                                              height=plot_height)
        self.cue_update_job()


@for_page(PAGE_SETTING_VIEW_DICT, PageSetting.BY_STIM_ID)
class ByStimIDView(ttk.Frame):

    def __init__(self: Self, parent: 'View', **kwargs) -> None:
        super().__init__(parent.container, **kwargs)
        self.parent = parent
        self.controller = None
        self.root_update_job = None
        self.root_updating_figure = False
        self.create_widgets()

    def create_widgets(self: Self) -> None:

        self.main_frame = ttk.Frame(self, style='Invisible.TFrame')
        self.main_frame.pack(padx=LARGE_PAD,
                             side='left',
                             fill='y',
                             expand=True)

        self.ci_frame = ttk.Frame(self.main_frame)
        self.ci_frame.pack(pady=LARGE_PAD, fill='both', expand=True)
        self.ci_heading = ttk.Label(self.ci_frame,
                                    justify='center',
                                    text='Calcium image',
                                    font=HEADING_FONT)
        self.ci_heading.pack(pady=(SMALL_PAD, 0))
        self.ci_label = ttk.Label(self.ci_frame, justify='center')
        self.ci_label.pack()
        self.left_button = ttk.Button(self.ci_frame, text='<<')
        self.left_button.pack(side='left', padx=SMALL_PAD)
        self.left_button.configure(command=self.on_left_button)
        self.right_button = ttk.Button(self.ci_frame, text='>>')
        self.right_button.pack(side='right', padx=SMALL_PAD)
        self.right_button.configure(command=self.on_right_button)
        self.tk_image = tk.Canvas(self.ci_frame)
        self.tk_image.pack(pady=SMALL_PAD)
        self.tk_image.bind('<Button>', self.on_image_click)
        self.tk_image_job = None
        self.tk_image_image = None
        self.image_original = None
        self.image_resized = None

        self.button_frame = ttk.Frame(self.ci_frame)
        self.button_frame.pack(pady=(0, SMALL_PAD))
        self.play_button = ttk.Button(self.button_frame, text='Play')
        self.play_button.pack(side='left', padx=(0, SMALL_PAD))
        self.play_button.configure(command=self.on_play_button(1))
        self.play_slow_button = ttk.Button(self.button_frame,
                                           text='Play (slow)')
        self.play_slow_button.pack(side='left', padx=(0, SMALL_PAD))
        self.play_slow_button.configure(
            command=self.on_play_button(SLOW_SPEED))
        self.stop_button = ttk.Button(self.button_frame, text='Stop')
        self.stop_button.pack(side='left', padx=(0, SMALL_PAD))
        self.stop_button.configure(command=self.on_stop_button)
        self.select_all_button = ttk.Button(self.button_frame,
                                            text='Select All')
        self.select_all_button.pack(side='left', padx=(0, SMALL_PAD))
        self.select_all_button.configure(command=self.on_select_all_button)
        self.jump_to_button = ttk.Button(self.button_frame, text='Jump to...')
        self.jump_to_button.pack(side='left', padx=(0, SMALL_PAD))
        self.jump_to_button.configure(command=self.on_jump_to_button)
        self.jump_to_var = tk.StringVar()
        self.jump_to_text_box = ttk.Entry(self.button_frame,
                                          textvariable=self.jump_to_var,
                                          width=5)
        self.jump_to_text_box.pack(side='left')

        self.button2_frame = ttk.Frame(self.ci_frame)
        self.button2_frame.pack(pady=(0, SMALL_PAD))
        self.start_from_odor_var = tk.BooleanVar()
        self.start_from_odor_button = ttk.Checkbutton(
            self.button2_frame,
            text='Play from odor presentation?',
            variable=self.start_from_odor_var)
        self.start_from_odor_button.pack(side='left', padx=(0, SMALL_PAD))
        self.plot_delta_var = tk.BooleanVar()
        self.plot_delta_button = ttk.Checkbutton(self.button2_frame,
                                                 text='Plot ΔF/F?',
                                                 variable=self.plot_delta_var)
        self.plot_delta_button.pack(side='left', padx=(0, SMALL_PAD))
        self.plot_delta_button.configure(command=self.cue_update_job_wrapper(
            update_figure=False))
        self.hide_rois_var = tk.BooleanVar()
        self.hide_rois_button = ttk.Checkbutton(self.button2_frame,
                                                text='Hide ROIs?',
                                                variable=self.hide_rois_var)
        self.hide_rois_button.pack(side='left', padx=(0, SMALL_PAD))
        self.hide_rois_button.configure(command=self.cue_update_job_wrapper(
            update_figure=False))
        self.save_tiff_button = ttk.Button(self.button2_frame,
                                           text='Save as TIFF')
        self.save_tiff_button.pack(side='left')
        self.save_tiff_button.configure(command=self.on_save_tiff_button)

        self.button3_frame = ttk.Frame(self.ci_frame)
        self.button3_frame.pack()
        self.spatial_blur_label = ttk.Label(self.button3_frame,
                                            text='Spatial blur:')
        self.spatial_blur_label.pack(side='left', padx=(0, SMALL_PAD))
        self.spatial_blur_var = tk.DoubleVar()
        self.spatial_blur_slider = tk.Scale(self.button3_frame,
                                            from_=0,
                                            to=3,
                                            resolution=0.1,
                                            orient='horizontal',
                                            variable=self.spatial_blur_var)
        self.spatial_blur_slider.pack(side='left', padx=(0, SMALL_PAD))
        self.spatial_blur_slider.configure(command=self.cue_update_job_wrapper(
            update_figure=False))
        self.temporal_blur_label = ttk.Label(self.button3_frame,
                                             text='Temporal blur:')
        self.temporal_blur_label.pack(side='left', padx=(0, SMALL_PAD))
        self.temporal_blur_var = tk.IntVar()
        self.temporal_blur_slider = tk.Scale(self.button3_frame,
                                             from_=0,
                                             to=10,
                                             orient='horizontal',
                                             variable=self.temporal_blur_var)
        self.temporal_blur_slider.pack(side='left', padx=(0, SMALL_PAD))
        self.temporal_blur_slider.configure(
            command=self.cue_update_job_wrapper(update_figure=False))

        self.side_frame = ttk.Frame(self, style='Invisible.TFrame')
        self.side_frame.pack(padx=(0, LARGE_PAD),
                             pady=(LARGE_PAD, LARGE_PAD),
                             side='right',
                             fill='y',
                             expand=True)

        self.display_frame = ttk.Frame(self.side_frame)
        self.display_frame.pack(fill='x')
        self.display_heading = ttk.Label(self.display_frame,
                                         text='Analysis',
                                         font=HEADING_FONT)
        self.display_heading.pack(pady=SMALL_PAD)
        self.fig = Figure()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.display_frame)
        self.canvas.get_tk_widget().pack()
        self.toolbar = NavigationToolbar2Tk(self.canvas,
                                            self.display_frame,
                                            pack_toolbar=False)
        self.toolbar.pack(pady=(0, SMALL_PAD), fill='x')
        self.plot_button_frame = FlowFrame(self.display_frame, SMALL_PAD)
        self.plot_button_frame.pack(fill='x', expand=True)
        self.plot_setting_var = tk.StringVar()
        self.plot_buttons = []
        for i, s in enumerate(ByStimIDPlotSetting):
            button = ttk.Radiobutton(self.plot_button_frame,
                                     text=s.value,
                                     variable=self.plot_setting_var,
                                     value=s.value)
            button.configure(command=self.on_plot_setting_button)
            self.plot_buttons.append(button)

        self.page_switcher_frame = ttk.Frame(self.side_frame)
        self.page_switcher_frame.pack(side='bottom', fill='x')
        self.page_switcher_label = ttk.Label(self.page_switcher_frame,
                                             text='View',
                                             font=HEADING_FONT)
        self.page_switcher_label.pack()
        self.page_button_frame = ttk.Frame(self.page_switcher_frame)
        self.page_button_frame.pack(pady=SMALL_PAD)
        self.page_buttons = []
        for i, s in enumerate(PageSetting):
            button = ttk.Radiobutton(self.page_button_frame,
                                     text=s.value,
                                     variable=self.parent.page_setting_var,
                                     value=s.value)
            button.pack(padx=int(SMALL_PAD / 2), side='left')
            button.configure(command=self.on_page_setting_button)
            self.page_buttons.append(button)

    @property
    def original_image_size(self: Self) -> tuple[int, int]:
        return self.image_original.size

    @property
    def displayed_image_size(self: Self) -> tuple[int, int]:
        return self.image_resized.size

    def set_ci_image(self: Self, caption: str, img: Image.Image) -> None:
        self.ci_label.configure(text=caption)
        self.image_original = img
        new_size = (self.tk_image.winfo_reqwidth(),
                    self.tk_image.winfo_reqheight())
        self.image_resized = img.resize(new_size)
        self.tk_image_image = ImageTk.PhotoImage(self.image_resized)
        self.tk_image.delete('IMG')
        self.tk_image.create_image(0,
                                   0,
                                   image=self.tk_image_image,
                                   anchor='nw',
                                   tags='IMG')

    def stop_image_job(self: Self) -> None:
        if self.tk_image_job is not None:
            self.tk_image.after_cancel(self.tk_image_job)
            self.tk_image_job = None
        self.controller.delete_running_lines()

    def show_in_progress_screen(self: Self) -> None:
        self.fig.clf()
        self.fig.text(0.5,
                      0.5,
                      'Computing...',
                      fontsize=30,
                      horizontalalignment='center',
                      verticalalignment='center')
        self.update_fig(flush=True)

    def update_fig(self: Self, flush=False) -> None:
        self.canvas.draw()
        if flush:
            self.canvas.flush_events()
        self.toolbar.update()

    def cue_update_job(self: Self, update_figure: bool = True) -> None:

        def _update():
            self.controller.update(redraw_figure=update_figure)
            self.root_update_job = None
            self.root_updating_figure = False

        if self.root_update_job is not None:
            self.after_cancel(self.root_update_job)
        self.root_updating_figure = self.root_updating_figure or update_figure
        self.root_update_job = self.after(JOB_SCHEDULE_DELAY, _update)

    def cue_update_job_wrapper(self: Self, update_figure: bool = True) -> None:

        def wrapper(*args: list[Any]):
            self.cue_update_job(update_figure=update_figure)

        return wrapper

    def on_left_button(self: Self) -> None:
        self.controller.decrement_stim_id()
        self.stop_image_job()
        self.controller.update()

    def on_right_button(self: Self) -> None:
        self.controller.increment_stim_id()
        self.stop_image_job()
        self.controller.update()

    def on_play_button(self: Self, speed: float) -> None:

        def wrapper() -> None:
            self.controller.play_ci_video(speed)

        return wrapper

    def on_stop_button(self: Self) -> None:
        self.stop_image_job()
        self.controller.update()

    def on_select_all_button(self: Self) -> None:
        self.stop_image_job()
        self.controller.select_all_rois()
        self.controller.update()

    def on_jump_to_button(self: Self) -> None:
        try:
            i = self.jump_to_var.get()
        except ValueError:
            return
        self.controller.jump_to_stim_id(int(i) - 1)
        self.stop_image_job()
        self.controller.update()

    def on_save_tiff_button(self: Self) -> None:
        self.controller.save_tiff()

    def on_image_click(self: Self, event: tk.Event) -> None:
        shift_pressed = (event.state == 1)
        orig_w, orig_h = self.original_image_size
        display_w, display_h = self.displayed_image_size
        x, y = (event.x / display_w) * orig_w, (event.y / display_h) * orig_h
        self.controller.process_image_click(shift_pressed, x, y)
        self.controller.update()

    def on_plot_setting_button(self: Self) -> None:
        self.stop_image_job()
        self.controller.update()

    def on_page_setting_button(self: Self) -> None:
        self.stop_image_job()
        self.parent.update_frame()
        self.controller.update()

    def on_resize_window(self: Self, width: int, height: int) -> None:
        ci_image_size = int(0.6 * min(width, height))
        self.tk_image.configure(width=ci_image_size, height=ci_image_size)
        plot_height = int(0.6 * height)
        plot_width = int(1.4 * plot_height)
        self.canvas.get_tk_widget().configure(width=plot_width,
                                              height=plot_height)
        self.cue_update_job()


class View(tk.Tk):

    def __init__(self: Self) -> None:
        super().__init__()

        self.window_width = None
        self.window_height = None
        self.controller = None

        self.title('Calcium imaging analyzer')
        self.configure(background='light grey')
        self.geometry(DEFAULT_WINDOW_SIZE)
        self.bind('<Configure>', self.on_resize_window)

        self.style = ttk.Style()
        self.style.configure('TFrame', background='light blue')
        self.style.configure('Invisible.TFrame', background='light grey')
        self.style.configure('TLabel', background='light blue')

        self.page_setting_var = tk.StringVar()
        self.children = dict()
        self.create_children()

    def create_children(self: Self) -> None:
        self.container = ttk.Frame(self, style='Invisible.TFrame')
        self.container.pack(fill='both', expand=True)
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)
        for s in PageSetting:
            frame = PAGE_SETTING_VIEW_DICT[s](self, style='Invisible.TFrame')
            self.children[s] = frame
            frame.grid(row=0, column=0, sticky='nsew')

    @property
    def page_setting(self: Self) -> PageSetting:
        return PageSetting(self.page_setting_var.get())

    @property
    def current_page(self: Self) -> ttk.Frame:
        return self.children[self.page_setting]

    def update_frame(self: Self) -> None:
        self.current_page.tkraise()
        if self.window_width is not None and self.window_height is not None:
            self.current_page.on_resize_window(self.window_width,
                                               self.window_height)
        self.current_page.controller.update()

    def on_resize_window(self: Self, event: tk.Event) -> None:
        if not self.page_setting_var.get():
            return
        if event.widget == self:
            w, h = event.width, event.height
            if (w != self.window_width) or (h != self.window_height):
                self.window_width = w
                self.window_height = h
                self.current_page.on_resize_window(w, h)
