from typing import Self

import numpy as np
from matplotlib.figure import Figure

from config import BASELINE_AVG_END, BASELINE_AVG_START
from datatypes import GUIStateMessage, PlotSetting, ROIManager


class PlotHandler:

    def __init__(self: Self) -> None:
        self.cache_line_behavioral_plot = None
        self.cache_line_fluorescence_plot = None

    def render_none_plot(self: Self, fig: Figure, message: GUIStateMessage,
                         frame: int | None) -> None:
        fig.clf()

    def render_behavioral_plot(self: Self, fig: Figure,
                               message: GUIStateMessage,
                               frame: int | None) -> None:
        metadata = message.metadata
        if frame is None:
            fig.clf()
            ax = fig.add_subplot()
            offset = metadata.frame_times[0]
            ax.plot(metadata.sniff_times - offset,
                    metadata.sniff_values,
                    label='Sniff')
            ax.axvline(metadata.odor_time - offset,
                       color='black',
                       label='Odor')
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
            if self.cache_line_behavioral_plot is not None:
                self.cache_line_behavioral_plot.remove()
            offset = metadata.frame_times[0]
            self.cache_line_behavioral_plot = fig.axes[0].axvline(
                metadata.frame_times[frame] - offset, color='red')

    def render_fluorescence_plot(self: Self, fig: Figure,
                                 message: GUIStateMessage,
                                 frame: int | None) -> None:
        metadata = message.metadata
        if frame is None:
            fig.clf()
            if message.rois_focused:
                ys, xs = np.indices(message.tiff_arr.shape[1:])
                ax = fig.add_subplot()
                offset = metadata.frame_times[0]
                for i, roi_name in enumerate(message.rois_focused):
                    indices_in_roi = ROIManager.is_in_roi(
                        xs, ys, message.rois[roi_name])
                    f_vals = np.mean(message.tiff_arr[:, indices_in_roi],
                                     axis=1)
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
                if self.cache_line_fluorescence_plot is not None:
                    self.cache_line_fluorescence_plot.remove()
                offset = metadata.frame_times[0]
                self.cache_line_fluorescence_plot = fig.axes[0].axvline(
                    metadata.frame_times[frame] - offset, color='red')

    def render_plot(self: Self, fig: Figure, message: GUIStateMessage,
                    frame: int | None) -> None:
        s = message.plot_setting
        if s == PlotSetting.NONE:
            self.render_none_plot(fig, message, frame)
        if s == PlotSetting.BEHAVIOR:
            self.render_behavioral_plot(fig, message, frame)
        elif s == PlotSetting.FLUORESCENCE:
            self.render_fluorescence_plot(fig, message, frame)

    def delete_running_lines(self: Self) -> None:
        if self.cache_line_behavioral_plot:
            self.cache_line_behavioral_plot.remove()
            self.cache_line_behavioral_plot = None
        if self.cache_line_fluorescence_plot:
            self.cache_line_fluorescence_plot.remove()
            self.cache_line_fluorescence_plot = None
