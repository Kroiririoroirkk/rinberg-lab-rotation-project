from __future__ import annotations

from typing import TYPE_CHECKING, Self

import numpy as np
from matplotlib.figure import Figure

from config import BASELINE_AVG_END, BASELINE_AVG_START
from datatypes import ByStimIDPlotSetting, ByTrialPlotSetting, ROIManager

if TYPE_CHECKING:
    from model import ByStimIDModel, ByTrialModel


class ByTrialPlotHandler:

    def __init__(self: Self) -> None:
        self.cache_line_behavioral_plot = None
        self.cache_line_fluorescence_plot = None

    def render_none_plot(self: Self, fig: Figure, m: ByTrialModel,
                         frame: int | None) -> None:
        fig.clf()
        fig.text(0.5,
                 0.5,
                 'Select a plot setting...',
                 fontsize=30,
                 horizontalalignment='center',
                 verticalalignment='center')

    def render_behavioral_plot(self: Self, fig: Figure, m: ByTrialModel,
                               frame: int | None) -> None:
        md = m.metadata
        if frame is None:
            fig.clf()
            ax = fig.add_subplot()
            offset = md.frame_times[0]
            ax.plot(md.sniff_times - offset, md.sniff_values, label='Sniff')
            ax.axvline(md.odor_time - offset, color='black', label='Odor')
            ax.axvspan(0,
                       md.frame_times[-1] - offset,
                       alpha=0.1,
                       color='red',
                       label='Video frames')
            max_val = np.max(md.sniff_values)
            ax.vlines(md.lick1_times - offset,
                      0,
                      max_val,
                      color='chocolate',
                      label='Lick left')
            ax.vlines(md.lick2_times - offset,
                      -max_val,
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
            offset = md.frame_times[0]
            self.cache_line_behavioral_plot = fig.axes[0].axvline(
                md.frame_times[frame] - offset, color='red')

    def render_fluorescence_plot(self: Self, fig: Figure, m: ByTrialModel,
                                 frame: int | None) -> None:
        md = m.metadata
        if frame is None:
            fig.clf()
            if m.rois_focused:
                ys, xs = np.indices(m.tiff_arr.shape[1:])
                ax = fig.add_subplot()
                offset = md.frame_times[0]
                for i, roi_name in enumerate(m.rois_focused):
                    indices_in_roi = ROIManager.is_in_roi(
                        xs, ys, m.rois[roi_name])
                    f_vals = np.mean(m.tiff_arr[:, indices_in_roi], axis=1)
                    median_f_val = np.median(
                        f_vals[BASELINE_AVG_START:BASELINE_AVG_END])
                    df_vals = (f_vals - median_f_val) / median_f_val
                    ax.plot(md.frame_times - offset,
                            df_vals,
                            label=f'ROI {i+1}')
                ax.axvline(md.odor_time - offset, color='black', label='Odor')
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
                offset = md.frame_times[0]
                self.cache_line_fluorescence_plot = fig.axes[0].axvline(
                    md.frame_times[frame] - offset, color='red')

    def render_plot(self: Self, fig: Figure, m: ByTrialModel,
                    frame: int | None) -> None:
        s = m.plot_setting
        if s == ByTrialPlotSetting.NONE:
            self.render_none_plot(fig, m, frame)
        if s == ByTrialPlotSetting.BEHAVIOR:
            self.render_behavioral_plot(fig, m, frame)
        elif s == ByTrialPlotSetting.FLUORESCENCE:
            self.render_fluorescence_plot(fig, m, frame)

    def delete_running_lines(self: Self) -> None:
        if self.cache_line_behavioral_plot:
            self.cache_line_behavioral_plot.remove()
            self.cache_line_behavioral_plot = None
        if self.cache_line_fluorescence_plot:
            self.cache_line_fluorescence_plot.remove()
            self.cache_line_fluorescence_plot = None


class ByStimIDPlotHandler:

    def __init__(self: Self) -> None:
        self.cache_line_fluorescence_plot = None

    def render_none_plot(self: Self, fig: Figure, m: ByStimIDModel,
                         frame: int | None) -> None:
        fig.clf()
        fig.text(0.5,
                 0.5,
                 'Select a plot setting...',
                 fontsize=30,
                 horizontalalignment='center',
                 verticalalignment='center')

    def render_fluorescence_plot(self: Self, fig: Figure, m: ByStimIDModel,
                                 frame: int | None) -> None:
        if frame is None:
            fig.clf()
            if m.rois_focused:
                ys, xs = np.indices(m.tiff_arr.shape[1:])
                ax = fig.add_subplot()
                for i, roi_name in enumerate(m.rois_focused):
                    indices_in_roi = ROIManager.is_in_roi(
                        xs, ys, m.rois[roi_name])
                    f_vals = np.mean(m.tiff_arr[:, indices_in_roi], axis=1)
                    median_f_val = np.median(
                        f_vals[BASELINE_AVG_START:BASELINE_AVG_END])
                    df_vals = (f_vals - median_f_val) / median_f_val
                    ax.plot(m.time_arr, df_vals, label=f'ROI {i+1}')
                ax.axvline(0, color='black', label='Odor')
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
                self.cache_line_fluorescence_plot = fig.axes[0].axvline(
                    m.time_arr[frame], color='red')

    def render_plot(self: Self, fig: Figure, m: ByStimIDModel,
                    frame: int | None) -> None:
        s = m.plot_setting
        if s == ByStimIDPlotSetting.NONE:
            self.render_none_plot(fig, m, frame)
        elif s == ByStimIDPlotSetting.FLUORESCENCE:
            self.render_fluorescence_plot(fig, m, frame)

    def delete_running_lines(self: Self) -> None:
        if self.cache_line_fluorescence_plot:
            self.cache_line_fluorescence_plot.remove()
            self.cache_line_fluorescence_plot = None
