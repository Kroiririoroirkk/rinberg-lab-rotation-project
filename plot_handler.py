from __future__ import annotations

from typing import TYPE_CHECKING, Self

import numpy as np
import scipy.stats
from matplotlib.figure import Figure

from config import (BASELINE_AVG_END, BASELINE_AVG_START, LATENCY_THRESHOLD,
                    NUM_FRAMES_AVG, NUM_FRAMES_BUYIN, ODOR_DICT)
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
        self.model = None
        self.mean_heatmap = None
        self.significance_baseline_heatmap = None
        self.significance_heatmap = None
        self.latency_heatmap = None
        self.peak_heatmap = None

    def process_tiffs(self: Self, m: ByStimIDModel) -> None:
        if self.model is m:
            return

        stim_conds = [m.stim_condition_dict[stim_id] for stim_id in m.stim_ids]
        num_rois = len(m.rois)
        num_conds = len(stim_conds)

        mean_heatmap = np.zeros((num_rois, num_conds))
        significance_heatmap = np.zeros((num_rois, num_conds))
        latency_heatmap = np.zeros((num_rois, num_conds))
        peak_heatmap = np.zeros((num_rois, num_conds))

        control = dict()

        for j in range(num_conds):
            print(f'Processing condition {j+1} out of {num_conds}...')
            tiff_arr, time_arr, _, individual_arrs = m.ci_handler.load_image(
                m, m.stim_ids[j], return_individual_images=True)
            ys, xs = np.indices(tiff_arr.shape[1:])
            tiff_arr_baseline = tiff_arr[BASELINE_AVG_START:BASELINE_AVG_END]
            odor_start_frame = np.argmax(time_arr > 0) + NUM_FRAMES_BUYIN
            odor_end_frame = odor_start_frame + NUM_FRAMES_AVG
            tiff_arr_odor = tiff_arr[odor_start_frame:odor_end_frame]

            # For calculating significance values
            i_tiff_arrs_baseline = []
            i_tiff_arrs_odor = []
            for i_tiff_arr in individual_arrs:
                i_tiff_arrs_baseline.append(
                    i_tiff_arr[BASELINE_AVG_START:BASELINE_AVG_END])
                i_tiff_arrs_odor.append(
                    i_tiff_arr[odor_start_frame:odor_end_frame])

            for i, roi_name in enumerate(m.rois):
                indices_in_roi = ROIManager.is_in_roi(xs, ys, m.rois[roi_name])

                f_vals_baseline = np.mean(tiff_arr_baseline[:, indices_in_roi],
                                          axis=1)
                median_f_val = np.median(f_vals_baseline)
                df_vals_baseline = (f_vals_baseline -
                                    median_f_val) / median_f_val
                f_vals_odor = np.mean(tiff_arr_odor[:, indices_in_roi], axis=1)
                df_vals_odor = (f_vals_odor - median_f_val) / median_f_val
                peak_df_odor = np.max(df_vals_odor)

                peak_heatmap[i, j] = peak_df_odor
                if np.any(df_vals_baseline > peak_df_odor * LATENCY_THRESHOLD):
                    latency_heatmap[i, j] = np.nan
                else:
                    crossing_idx = np.argmax(df_vals_odor > peak_df_odor *
                                             LATENCY_THRESHOLD)
                    latency_heatmap[i, j] = time_arr[odor_start_frame +
                                                     crossing_idx]
                mean_heatmap[i, j] = np.mean(df_vals_odor)

                # For calculating significance values
                df_val_odor_means = []
                for b, o in zip(i_tiff_arrs_baseline, i_tiff_arrs_odor):
                    i_f_vals_baseline = np.mean(b[:, indices_in_roi], axis=1)
                    i_median_f_val = np.median(i_f_vals_baseline)
                    i_f_vals_odor = np.mean(o[:, indices_in_roi], axis=1)
                    i_df_vals_odor = (i_f_vals_odor -
                                      i_median_f_val) / i_median_f_val
                    i_df_val_odor_mean = np.mean(i_df_vals_odor)
                    df_val_odor_means.append(i_df_val_odor_mean)
                if roi_name not in control:
                    control[roi_name] = df_val_odor_means
                _, pval = scipy.stats.mannwhitneyu(control[roi_name],
                                                   df_val_odor_means)
                significance_heatmap[i, j] = pval

        self.tiffs_processed = True
        self.mean_heatmap = mean_heatmap
        self.significance_heatmap = significance_heatmap
        self.latency_heatmap = latency_heatmap
        self.peak_heatmap = peak_heatmap
        self.model = m

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

    def generate_conds_from_odor(
            self: Self, m: ByStimIDModel,
            odor: int | None) -> tuple[list[int], list[str]]:
        stim_conds = [m.stim_condition_dict[stim_id] for stim_id in m.stim_ids]
        if odor == 1:
            stim_cond_idx = [
                i for i, cond in enumerate(stim_conds)
                if cond.odor2 == Odor.EMPTY
            ]
            odor_vals = [(stim_conds[i].odor1, stim_conds[i].odor1_flow, i)
                         for i in stim_cond_idx]
            odors = list(ODOR_DICT.values())
            odor_vals.sort(key=lambda t: (odors.index(t[0]), t[1]))
            stim_cond_idx = [i for _, _, i in odor_vals]
            odor_labels = [
                f'{m.stim_ids[i]}: {odor_flow} {odor}'
                for odor, odor_flow, i in odor_vals
            ]
        elif odor == 2:
            stim_cond_idx = [
                i for i, cond in enumerate(stim_conds)
                if cond.odor1 == Odor.EMPTY
            ]
            odor_vals = [(stim_conds[i].odor2, stim_conds[i].odor2_flow, i)
                         for i in stim_cond_idx]
            odors = list(ODOR_DICT.values())
            odor_vals.sort(key=lambda t: (odors.index(t[0]), t[1]))
            stim_cond_idx = [i for _, _, i in odor_vals]
            odor_labels = [
                f'{m.stim_ids[i]}: {odor_flow} {odor}'
                for odor, odor_flow, i in odor_vals
            ]
        elif odor is None:
            stim_cond_idx = list(range(len(stim_conds)))
            odor_vals = [(stim_conds[i].odor1, stim_conds[i].odor1_flow,
                          stim_conds[i].odor2, stim_conds[i].odor2_flow, i)
                         for i in stim_cond_idx]
            odors = list(ODOR_DICT.values())
            odor_vals.sort(key=lambda t:
                           (odors.index(t[0]), t[1], odors.index(t[2]), t[3]))
            stim_cond_idx = [i for _, _, _, _, i in odor_vals]
            odor_labels = [
                f'{m.stim_ids[i]}: {odor_flow1} {odor1} + {odor_flow2} {odor2}'
                for odor1, odor_flow1, odor2, odor_flow2, i in odor_vals
            ]
        else:
            raise ValueError(
                f'Odor {odor} is not valid (needs to be either 1, 2, or None)')

        return stim_cond_idx, odor_labels

    def render_odor_response_plot(self: Self, fig: Figure, m: ByStimIDModel,
                                  frame: int | None, odor: int) -> None:
        if frame is not None:
            return
        fig.clf()
        if m.rois_focused:
            stim_cond_idx, odor_labels = self.generate_conds_from_odor(m, odor)

            if len(stim_cond_idx) == 0:
                fig.text(0.5,
                         0.5,
                         f'No pure odor {odor} trials to plot.',
                         fontsize=30,
                         horizontalalignment='center',
                         verticalalignment='center')
                return

            roi_idx = [
                list(m.rois).index(roi_name) for roi_name in m.rois_focused
            ]
            num_rois = len(m.rois_focused)

            x_labels = odor_labels
            y_labels = [f'ROI {i+1}' for i in range(num_rois)]

            self.process_tiffs(m)
            idx = np.ix_(roi_idx, stim_cond_idx)
            heatmap = self.mean_heatmap[idx]
            significance_heatmap = self.significance_heatmap[idx]

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
                    if significance_heatmap[i, j] < 0.05:
                        ax.text(j,
                                i,
                                '*',
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
            stim_cond_idx, odor_labels = self.generate_conds_from_odor(m, odor)

            if len(stim_cond_idx) == 0:
                fig.text(0.5,
                         0.5,
                         f'No pure odor {odor} trials to plot.',
                         fontsize=30,
                         horizontalalignment='center',
                         verticalalignment='center')
                return

            roi_idx = [
                list(m.rois).index(roi_name) for roi_name in m.rois_focused
            ]
            num_rois = len(m.rois_focused)

            x_labels = odor_labels
            y_labels = [f'ROI {i+1}' for i in range(num_rois)]

            self.process_tiffs(m)
            idx = np.ix_(roi_idx, stim_cond_idx)
            heatmap = self.latency_heatmap[idx]
            significance_heatmap = self.significance_heatmap[idx]

            ax = fig.add_subplot()
            im = ax.imshow(heatmap, aspect='auto')
            fig.colorbar(
                im, label=f'Latency to {LATENCY_THRESHOLD:.2f} × max (ms)')
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
                    if significance_heatmap[i, j] < 0.05:
                        ax.text(j,
                                i,
                                '*',
                                horizontalalignment='center',
                                verticalalignment='center',
                                color='w')
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
            stim_cond_idx, odor_labels = self.generate_conds_from_odor(m, None)
            roi_idx = [
                list(m.rois).index(roi_name) for roi_name in m.rois_focused
            ]
            num_rois = len(m.rois_focused)
            x_labels = odor_labels
            y_labels = [f'ROI {i+1}' for i in range(num_rois)]

            self.process_tiffs(m)
            idx = np.ix_(roi_idx, stim_cond_idx)
            heatmap = self.peak_heatmap[idx]
            significance_heatmap = self.significance_heatmap[idx]

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

    def render_num_gloms_active_plot(self: Self, fig: Figure, m: ByStimIDModel,
                                     frame: int | None) -> None:
        if frame is not None:
            return
        fig.clf()
        if m.rois_focused:
            stim_cond_idx, odor_labels = self.generate_conds_from_odor(m, None)
            roi_idx = [
                list(m.rois).index(roi_name) for roi_name in m.rois_focused
            ]
            x_labels = odor_labels

            self.process_tiffs(m)
            idx = np.ix_(roi_idx, stim_cond_idx)
            significance_heatmap = self.significance_heatmap[idx]
            heatmap = significance_heatmap < 0.05
            num_gloms_active = np.sum(heatmap, axis=0)

            ax = fig.add_subplot()
            ax.bar(range(len(x_labels)), num_gloms_active)
            ax.set_xlabel('Stimulus condition')
            ax.set_xticks(range(len(x_labels)),
                          labels=x_labels,
                          rotation=45,
                          horizontalalignment='right',
                          rotation_mode='anchor')
            ax.set_ylabel('# of ROIs significantly active (p < 0.05)')
            ax.set_title('Number of glomeruli active by stimulus condition')
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
        elif s == ByStimIDPlotSetting.NUM_GLOMS_ACTIVE:
            self.render_num_gloms_active_plot(fig, m, frame)

    def delete_running_lines(self: Self) -> None:
        if self.cache_line_fluorescence_plot:
            try:
                self.cache_line_fluorescence_plot.remove()
            except NotImplementedError:
                pass
            self.cache_line_fluorescence_plot = None
