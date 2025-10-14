from __future__ import annotations

from typing import TYPE_CHECKING, Self

import numpy as np
import scipy.stats
from matplotlib.figure import Figure

from config import (BASELINE_AVG_END, BASELINE_AVG_START, NUM_FRAMES_AVG,
                    NUM_FRAMES_BUYIN)
from datatypes import ByStimIDPlotSetting, ByTrialPlotSetting, Odor, ROIManager

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
                try:
                    self.cache_line_behavioral_plot.remove()
                except NotImplementedError:
                    pass
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
                    try:
                        self.cache_line_fluorescence_plot.remove()
                    except NotImplementedError:
                        pass
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
            try:
                self.cache_line_behavioral_plot.remove()
            except NotImplementedError:
                pass
            self.cache_line_behavioral_plot = None
        if self.cache_line_fluorescence_plot:
            try:
                self.cache_line_fluorescence_plot.remove()
            except NotImplementedError:
                pass
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
                    try:
                        self.cache_line_fluorescence_plot.remove()
                    except NotImplementedError:
                        pass
                self.cache_line_fluorescence_plot = fig.axes[0].axvline(
                    m.time_arr[frame], color='red')

    def render_odor_response_plot(self: Self, fig: Figure, m: ByStimIDModel,
                                  frame: int | None, odor: int) -> None:
        if frame is not None:
            return
        fig.clf()
        if m.rois_focused:
            stim_conds = [
                m.stim_condition_dict[stim_id] for stim_id in m.stim_ids
            ]
            if odor == 1:
                stim_cond_idx = [
                    i for i, cond in enumerate(stim_conds)
                    if cond.odor2 == Odor.EMPTY
                ]
                odor_vals = [
                    0 if stim_conds[i].odor1 == Odor.EMPTY else
                    stim_conds[i].odor1_flow for i in stim_cond_idx
                ]
                sort_idx = np.argsort(odor_vals)
                stim_cond_idx = [stim_cond_idx[i] for i in sort_idx]
                odor_vals = [odor_vals[i] for i in sort_idx]
            elif odor == 2:
                stim_cond_idx = [
                    i for i, cond in enumerate(stim_conds)
                    if cond.odor1 == Odor.EMPTY
                ]
                odor_vals = [
                    0 if stim_conds[i].odor2 == Odor.EMPTY else
                    stim_conds[i].odor2_flow for i in stim_cond_idx
                ]
                sort_idx = np.argsort(odor_vals)
                stim_cond_idx = [stim_cond_idx[i] for i in sort_idx]
                odor_vals = [odor_vals[i] for i in sort_idx]
            else:
                raise ValueError(
                    f'Odor {odor} is not valid (needs to be either 1 or 2)')

            if len(stim_cond_idx) == 0:
                fig.text(0.5,
                         0.5,
                         f'No pure odor {odor} trials to plot.',
                         fontsize=30,
                         horizontalalignment='center',
                         verticalalignment='center')
                return

            num_rois = len(m.rois_focused)

            x_labels = odor_vals
            y_labels = [f'ROI {i+1}' for i in range(num_rois)]
            heatmap = np.zeros((num_rois, len(x_labels)))
            significance_heatmap = np.zeros_like(heatmap)

            for i, roi_name in enumerate(m.rois_focused):
                for rel_j, j in enumerate(stim_cond_idx):
                    tiff_arr, time_arr, _ = m.load_tiff_arr(j)
                    ys, xs = np.indices(tiff_arr.shape[1:])
                    indices_in_roi = ROIManager.is_in_roi(
                        xs, ys, m.rois[roi_name])

                    tiff_arr_baseline = tiff_arr[
                        BASELINE_AVG_START:BASELINE_AVG_END]
                    odor_start_frame = np.argmax(
                        time_arr > 0) + NUM_FRAMES_BUYIN
                    odor_end_frame = odor_start_frame + NUM_FRAMES_AVG
                    tiff_arr_odor = tiff_arr[odor_start_frame:odor_end_frame]

                    f_vals_baseline = np.mean(
                        tiff_arr_baseline[:, indices_in_roi], axis=1)
                    median_f_val = np.median(f_vals_baseline)
                    f_vals_odor = np.mean(tiff_arr_odor[:, indices_in_roi],
                                          axis=1)
                    df_vals = (f_vals_odor - median_f_val) / median_f_val
                    heatmap[i, rel_j] = np.mean(df_vals)
                    _, pval = scipy.stats.mannwhitneyu(f_vals_baseline,
                                                       f_vals_odor)
                    significance_heatmap[i, rel_j] = pval

            ax = fig.add_subplot()
            im = ax.imshow(heatmap, aspect='auto')
            fig.colorbar(im, label='Average ΔF/F (odor period)')
            ax.set_xlabel(f'Odor {odor} flow')
            ax.set_xticks(range(len(x_labels)),
                          labels=x_labels,
                          rotation=45,
                          horizontalalignment='right',
                          rotation_mode='anchor')
            ax.set_ylabel('ROI')
            ax.set_yticks(range(len(y_labels)), labels=y_labels)
            for i in range(len(y_labels)):
                for j in range(len(x_labels)):
                    ax.text(j,
                            i,
                            f'p={significance_heatmap[i, j]:.3f}',
                            horizontalalignment='center',
                            verticalalignment='center',
                            color='w')
            ax.set_title(f'Glomerulus activity by concentration (odor {odor})')
            fig.tight_layout()
        else:
            fig.text(0.5,
                     0.5,
                     'Select an ROI...',
                     fontsize=30,
                     horizontalalignment='center',
                     verticalalignment='center')

    def render_odor_1_response_plot(self: Self, fig: Figure, m: ByStimIDModel,
                                    frame: int | None) -> None:
        return self.render_odor_response_plot(fig, m, frame, 1)

    def render_odor_2_response_plot(self: Self, fig: Figure, m: ByStimIDModel,
                                    frame: int | None) -> None:
        return self.render_odor_response_plot(fig, m, frame, 2)

    def render_odor_latency_plot(self: Self, fig: Figure, m: ByStimIDModel,
                                 frame: int | None, odor: int) -> None:
        if frame is not None:
            return
        fig.clf()
        if m.rois_focused:
            stim_conds = [
                m.stim_condition_dict[stim_id] for stim_id in m.stim_ids
            ]
            if odor == 1:
                stim_cond_idx = [
                    i for i, cond in enumerate(stim_conds)
                    if cond.odor2 == Odor.EMPTY
                ]
                odor_vals = [
                    0 if stim_conds[i].odor1 == Odor.EMPTY else
                    stim_conds[i].odor1_flow for i in stim_cond_idx
                ]
                sort_idx = np.argsort(odor_vals)
                stim_cond_idx = [stim_cond_idx[i] for i in sort_idx]
                odor_vals = [odor_vals[i] for i in sort_idx]
            elif odor == 2:
                stim_cond_idx = [
                    i for i, cond in enumerate(stim_conds)
                    if cond.odor1 == Odor.EMPTY
                ]
                odor_vals = [
                    0 if stim_conds[i].odor2 == Odor.EMPTY else
                    stim_conds[i].odor2_flow for i in stim_cond_idx
                ]
                sort_idx = np.argsort(odor_vals)
                stim_cond_idx = [stim_cond_idx[i] for i in sort_idx]
                odor_vals = [odor_vals[i] for i in sort_idx]
            else:
                raise ValueError(
                    f'Odor {odor} is not valid (needs to be either 1 or 2)')

            if len(stim_cond_idx) == 0:
                fig.text(0.5,
                         0.5,
                         f'No pure odor {odor} trials to plot.',
                         fontsize=30,
                         horizontalalignment='center',
                         verticalalignment='center')
                return

            num_rois = len(m.rois_focused)

            x_labels = odor_vals
            y_labels = [f'ROI {i+1}' for i in range(num_rois)]
            heatmap = np.zeros((num_rois, len(x_labels)))

            for i, roi_name in enumerate(m.rois_focused):
                for rel_j, j in enumerate(stim_cond_idx):
                    tiff_arr, time_arr, _ = m.load_tiff_arr(j)
                    ys, xs = np.indices(tiff_arr.shape[1:])
                    indices_in_roi = ROIManager.is_in_roi(
                        xs, ys, m.rois[roi_name])

                    tiff_arr_baseline = tiff_arr[
                        BASELINE_AVG_START:BASELINE_AVG_END]
                    odor_start_frame = np.argmax(
                        time_arr > 0) + NUM_FRAMES_BUYIN
                    odor_end_frame = odor_start_frame + NUM_FRAMES_AVG
                    tiff_arr_odor = tiff_arr[odor_start_frame:odor_end_frame]

                    f_vals_baseline = np.mean(
                        tiff_arr_baseline[:, indices_in_roi], axis=1)
                    median_f_val = np.median(f_vals_baseline)
                    df_baseline = (f_vals_baseline -
                                   median_f_val) / median_f_val
                    f_vals_odor = np.mean(tiff_arr_odor[:, indices_in_roi],
                                          axis=1)
                    df_vals = (f_vals_odor - median_f_val) / median_f_val
                    peak_df = np.max(df_vals)
                    if np.any(df_baseline > peak_df / 2):
                        heatmap[i, rel_j] = np.nan
                    else:
                        cross_frame = np.argmax(df_vals > peak_df / 2)
                        heatmap[i, rel_j] = time_arr[odor_start_frame +
                                                     cross_frame]

            ax = fig.add_subplot()
            im = ax.imshow(heatmap, aspect='auto')
            fig.colorbar(im, label='Latency to half-max (ms)')
            ax.set_xlabel(f'Odor {odor} flow')
            ax.set_xticks(range(len(x_labels)),
                          labels=x_labels,
                          rotation=45,
                          horizontalalignment='right',
                          rotation_mode='anchor')
            ax.set_ylabel('ROI')
            ax.set_yticks(range(len(y_labels)), labels=y_labels)
            ax.set_title(f'Glomerulus latency by concentration (odor {odor})')
            fig.tight_layout()
        else:
            fig.text(0.5,
                     0.5,
                     'Select an ROI...',
                     fontsize=30,
                     horizontalalignment='center',
                     verticalalignment='center')

    def render_odor_1_latency_plot(self: Self, fig: Figure, m: ByStimIDModel,
                                   frame: int | None) -> None:
        return self.render_odor_latency_plot(fig, m, frame, 1)

    def render_odor_2_latency_plot(self: Self, fig: Figure, m: ByStimIDModel,
                                   frame: int | None) -> None:
        return self.render_odor_latency_plot(fig, m, frame, 2)

    def render_glom_max_response_plot(self: Self, fig: Figure,
                                      m: ByStimIDModel,
                                      frame: int | None) -> None:
        if frame is not None:
            return
        fig.clf()
        if m.rois_focused:
            num_rois = len(m.rois_focused)
            x_labels = m.stim_ids
            y_labels = [f'ROI {i+1}' for i in range(num_rois)]
            heatmap = np.zeros((num_rois, len(x_labels)))
            significance_heatmap = np.zeros_like(heatmap)

            for i, roi_name in enumerate(m.rois_focused):
                for j, stim_id in enumerate(m.stim_ids):
                    tiff_arr, time_arr, _ = m.load_tiff_arr(j)
                    ys, xs = np.indices(tiff_arr.shape[1:])
                    indices_in_roi = ROIManager.is_in_roi(
                        xs, ys, m.rois[roi_name])

                    tiff_arr_baseline = tiff_arr[
                        BASELINE_AVG_START:BASELINE_AVG_END]
                    odor_start_frame = np.argmax(
                        time_arr > 0) + NUM_FRAMES_BUYIN
                    odor_end_frame = odor_start_frame + NUM_FRAMES_AVG
                    tiff_arr_odor = tiff_arr[odor_start_frame:odor_end_frame]

                    f_vals_baseline = np.mean(
                        tiff_arr_baseline[:, indices_in_roi], axis=1)
                    median_f_val = np.median(f_vals_baseline)
                    f_vals_odor = np.mean(tiff_arr_odor[:, indices_in_roi],
                                          axis=1)
                    df_vals = (f_vals_odor - median_f_val) / median_f_val
                    peak_df = np.max(df_vals)
                    heatmap[i, j] = peak_df
                    _, pval = scipy.stats.mannwhitneyu(f_vals_baseline,
                                                       f_vals_odor)
                    significance_heatmap[i, j] = pval

            ax = fig.add_subplot()
            im = ax.imshow(heatmap, aspect='auto')
            fig.colorbar(im, label='Peak ΔF/F (odor period)')
            ax.set_xlabel('Stimulus condition')
            ax.set_xticks(range(len(x_labels)),
                          labels=x_labels,
                          rotation=45,
                          horizontalalignment='right',
                          rotation_mode='anchor')
            ax.set_ylabel('ROI')
            ax.set_yticks(range(len(y_labels)), labels=y_labels)
            for i in range(len(y_labels)):
                for j in range(len(x_labels)):
                    if significance_heatmap[i, j] < 0.05:
                        ax.text(j,
                                i,
                                '*',
                                horizontalalignment='center',
                                verticalalignment='center',
                                color='w')
            ax.set_title('Peak glomerular responses by stimulus condition')
            fig.tight_layout()
        else:
            fig.text(0.5,
                     0.5,
                     'Select an ROI...',
                     fontsize=30,
                     horizontalalignment='center',
                     verticalalignment='center')

    def render_plot(self: Self, fig: Figure, m: ByStimIDModel,
                    frame: int | None) -> None:
        s = m.plot_setting
        if s == ByStimIDPlotSetting.NONE:
            self.render_none_plot(fig, m, frame)
        elif s == ByStimIDPlotSetting.FLUORESCENCE:
            self.render_fluorescence_plot(fig, m, frame)
        elif s == ByStimIDPlotSetting.ODOR_1_RESPONSE:
            self.render_odor_1_response_plot(fig, m, frame)
        elif s == ByStimIDPlotSetting.ODOR_2_RESPONSE:
            self.render_odor_2_response_plot(fig, m, frame)
        elif s == ByStimIDPlotSetting.ODOR_1_LATENCY:
            self.render_odor_1_latency_plot(fig, m, frame)
        elif s == ByStimIDPlotSetting.ODOR_2_LATENCY:
            self.render_odor_2_latency_plot(fig, m, frame)
        elif s == ByStimIDPlotSetting.GLOM_MAX_RESPONSE:
            self.render_glom_max_response_plot(fig, m, frame)

    def delete_running_lines(self: Self) -> None:
        if self.cache_line_fluorescence_plot:
            try:
                self.cache_line_fluorescence_plot.remove()
            except NotImplementedError:
                pass
            self.cache_line_fluorescence_plot = None
