from matplotlib.figure import Figure
from datatypes import TrialMetadata, ROIManager, ROIName, PlotSetting
import numpy as np
from numpy.typing import NDArray
from config import BASELINE_AVG_START, BASELINE_AVG_END


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
                             tiff_arr: NDArray[np.int16], rois: ROIManager,
                             rois_focused: list[ROIName],
                             frame: int | None) -> None:
    if frame is None:
        fig.clf()
        if rois_focused:
            ys, xs = np.indices(tiff_arr.shape[1:])
            ax = fig.add_subplot()
            offset = metadata.frame_times[0]
            for i, roi_name in enumerate(rois_focused):
                indices_in_roi = ROIManager.is_in_roi(xs, ys, rois[roi_name])
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
                rois: ROIManager, rois_focused: list[ROIName],
                frame: int | None) -> None:
    if plot_setting == PlotSetting.NONE:
        render_none_plot(fig)
    if plot_setting == PlotSetting.BEHAVIOR:
        render_behavioral_plot(fig, metadata, frame)
    elif plot_setting == PlotSetting.FLUORESCENCE:
        render_fluorescence_plot(fig, metadata, tif_arr, rois, rois_focused,
                                 frame)


def delete_running_lines() -> None:
    global cache_line_behavioral_plot, cache_line_fluorescence_plot
    if cache_line_behavioral_plot:
        cache_line_behavioral_plot.remove()
        cache_line_behavioral_plot = None
    if cache_line_fluorescence_plot:
        cache_line_fluorescence_plot.remove()
        cache_line_fluorescence_plot = None
