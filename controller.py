from typing import Self

import numpy as np

from datatypes import PlotSetting, ROIManager, ROIName

from model import Model
from view import View


class Controller:

    def __init__(self: Self, model: Model, view: View):
        self.model = model
        self.view = view
        self.view.set_controller(self)
        self.initialize_view()

    def update(self: Self,
               redraw_image: bool = True,
               redraw_figure: bool = True,
               frame: int | None = None) -> None:
        self.update_model_vars()
        if redraw_image:
            caption, image = self.model.update_ci(frame)
            self.view.ci_label.configure(text=caption)
            self.view.set_ci_image(image)
        if redraw_figure:
            self.model.update_plot(self.view.fig, frame)
            self.view.canvas.draw()
            self.view.toolbar.update()

    def decrement_tiff_file(self: Self) -> None:
        self.model.tiff_file_i = self.model.tiff_file_i - 1
        self.view.stop_image_job()

    def increment_tiff_file(self: Self) -> None:
        self.model.tiff_file_i = self.model.tiff_file_i + 1
        self.view.stop_image_job()

    def play_ci_video(self: Self, speed: float) -> None:

        def render(i: int) -> None:
            if i + 1 < self.model.tiff_arr.shape[0]:
                self.update(frame=i)
                dt = int((self.model.metadata.frame_times[i + 1] -
                          self.model.metadata.frame_times[i]) / speed)
                self.view.tk_image_job = self.view.tk_image.after(
                    dt if dt > 0 else 0, lambda: render(i + 1))
            else:
                self.view.stop_image_job()
                self.update()

        self.view.stop_image_job()

        if self.view.start_from_odor_var.get():
            i = np.argmax(self.model.metadata.frame_times >
                          self.model.metadata.odor_time)
            render(i - 5 if i > 5 else 0)
        else:
            render(0)

    def select_all_rois(self: Self) -> None:
        self.model.rois_focused = list(self.model.rois.keys())

    def jump_to_frame(self: Self, i: int) -> None:
        self.model.tiff_file_i = i

    def save_tiff(self: Self) -> None:
        self.model.export_tiff()

    def process_image_click(self: Self, shift_pressed: bool, x: int,
                            y: int) -> None:
        roi_name = None
        for roi_n, roi in self.model.rois.items():
            if ROIManager.is_in_roi(x, y, roi):
                roi_name = roi_n
        self.on_roi_click(roi_name, shift_pressed)

    def on_roi_click(self: Self, roi_name: ROIName | None,
                     shift_pressed: bool) -> None:
        self.view.stop_image_job()
        if shift_pressed:
            if roi_name is not None:
                self.model.toggle_roi(roi_name)
        else:
            if roi_name is None:
                self.model.rois_focused = []
            else:
                self.model.rois_focused = [roi_name]

    def delete_running_lines(self: Self) -> None:
        self.model.delete_running_lines()

    def initialize_view(self: Self) -> None:
        self.view.start_from_odor_var.set(self.model.start_from_odor)
        self.view.hide_rois_var.set(self.model.hide_rois)
        self.view.plot_delta_var.set(self.model.plot_delta)
        self.view.spatial_blur_var.set(self.model.spatial_blur)
        self.view.temporal_blur_var.set(self.model.temporal_blur)
        self.view.plot_setting_var.set(self.model.plot_setting.value)

    def update_model_vars(self: Self) -> None:
        self.model.start_from_odor = self.view.start_from_odor_var.get()
        self.model.hide_rois = self.view.hide_rois_var.get()
        self.model.plot_delta = self.view.plot_delta_var.get()
        self.model.spatial_blur = self.view.spatial_blur_var.get()
        self.model.temporal_blur = self.view.temporal_blur_var.get()
        self.model.plot_setting = PlotSetting(self.view.plot_setting_var.get())
