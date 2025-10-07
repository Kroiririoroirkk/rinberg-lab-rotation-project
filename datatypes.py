from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Self, NewType

from collections import OrderedDict
import read_roi
import h5py
import numpy as np
from numpy.typing import NDArray


class Odor(Enum):
    ETHYL_TIGLATE: Self = 'Ethyl Tiglate'
    TMBA: Self = '2MBA'
    EMPTY: Self = 'Empty'

    @staticmethod
    def lookup(s: bytes) -> Self | None:
        d = {
            b'EthylTiglate': Odor.ETHYL_TIGLATE,
            b'2MBA': Odor.TMBA,
            b'empty': Odor.EMPTY
        }
        return d.get(s)


class Choice(Enum):
    LEFT: Self = 'Left'
    RIGHT: Self = 'Right'
    MISS: Self = 'Miss'

    @staticmethod
    def lookup_target(i: int) -> Self | None:
        d = {0: Choice.LEFT, 1: Choice.RIGHT}
        return d.get(i)

    @staticmethod
    def lookup_response(i: int) -> Self | None:
        d = {
            1: Choice.RIGHT,
            2: Choice.LEFT,
            3: Choice.LEFT,
            4: Choice.RIGHT,
            5: Choice.MISS,
            6: Choice.MISS
        }
        return d.get(i)


def _read_clumped_arr(arr: h5py._hl.dataset.Dataset) -> list[int]:
    return np.array([int(x) for x in np.concatenate(arr)
                     ]) if arr.size else np.array([])


@dataclass
class TrialMetadata:
    frame_times: NDArray[int]  # in ms
    odor_time: int  # in ms
    lick1_times: NDArray[int]  # in ms
    lick2_times: NDArray[int]  # in ms
    sniff_times: NDArray[int]  # in ms
    sniff_values: NDArray[int]  # in relative units
    odor1: Odor
    odor2: Odor
    odor1_flow: float
    odor2_flow: float
    target: Choice
    response: Choice
    correct: bool

    @staticmethod
    def list_from_h5(h5_path: Path) -> list[Self]:
        trial_metadata_list = []
        f = h5py.File(h5_path, 'r')
        all_trials_f = f['Trials']
        recorded_trials = np.nonzero(all_trials_f['record'])[0]
        for i in recorded_trials:
            trial_f = f[f'Trial{i+1:04d}']
            frame_times = _read_clumped_arr(trial_f['frame_triggers'])
            odor_time = int(all_trials_f['fvOnTime_1st'][i])
            lick1_times = _read_clumped_arr(trial_f['lick1'])
            lick2_times = _read_clumped_arr(trial_f['lick2'])
            sniff_chunk_len1 = [len(s) for s in trial_f['sniff']]
            sniff_chunk_len2 = [
                int(s) for s in trial_f['Events']['sniff_samples']
            ]
            sniff_chunk_len2 = [s for s in sniff_chunk_len2 if s != 0]
            if sniff_chunk_len1 != sniff_chunk_len2:
                raise ValueError(
                    f'Sniff chunk lengths not consistent for Trial{i+1:04d}: '
                    f'{sniff_chunk_len1} != {sniff_chunk_len2}.')
            sniff_times = []
            for j in range(trial_f['Events'].shape[0] - 1):
                chunk_start_time, chunk_size = trial_f['Events'][j]
                chunk_end_time, _ = trial_f['Events'][j + 1]
                sniff_times.extend([
                    int(x) for x in np.linspace(chunk_start_time,
                                                chunk_end_time,
                                                chunk_size,
                                                endpoint=False)
                ])
            sniff_times = np.array(sniff_times)
            sniff_values = _read_clumped_arr(trial_f['sniff'][:-1])
            if sniff_times.size != sniff_values.size:
                raise ValueError(
                    'Sniff time and sniff value lists do not match up.')
            o1, o2 = all_trials_f['olfa_1st_0_odor'][i], all_trials_f[
                'olfa_1st_1_odor'][i]
            odor1, odor2 = Odor.lookup(o1), Odor.lookup(o2)
            if odor1 is None or odor2 is None:
                raise ValueError(f'Odor lookup failed on odors: {o1}, {o2}.')
            odor1_flow = float(all_trials_f['olfa_1st_0_mfc_1_flow'][i])
            odor2_flow = float(all_trials_f['olfa_1st_1_mfc_1_flow'][i])
            t = all_trials_f['trialtype'][i]
            target = Choice.lookup_target(t)
            if target is None:
                raise ValueError(f'Target lookup failed on input: {t}.')
            r = all_trials_f['result'][i]
            response = Choice.lookup_response(r)
            if response is None:
                raise ValueError(f'Response lookup failed on input: {r}.')
            correct = (target == response)
            trial_metadata_list.append(
                TrialMetadata(frame_times=frame_times,
                              odor_time=odor_time,
                              lick1_times=lick1_times,
                              lick2_times=lick2_times,
                              sniff_times=sniff_times,
                              sniff_values=sniff_values,
                              odor1=odor1,
                              odor2=odor2,
                              odor1_flow=odor1_flow,
                              odor2_flow=odor2_flow,
                              target=target,
                              response=response,
                              correct=correct))
        return trial_metadata_list


ROIName = NewType('ROIName', str)
ROI = NewType('ROI', dict[str, str | int])
ROIManager = NewType('ROIManager', OrderedDict[ROIName, ROI])


def _from_zip(roi_zip_path: Path) -> ROIManager:
    rois = read_roi.read_roi_zip(roi_zip_path)
    for v in rois.values():
        if v.get('type') != 'oval':
            raise ValueError('Cannot process non-oval ROI.')
    return ROIManager({ROIName(k): ROI(v) for k, v in rois.items()})


def _is_in_roi(x: int, y: int, roi: ROI) -> bool:
    half_w, half_h = roi['width'] / 2, roi['height'] / 2
    x_cent, y_cent = roi['left'] + half_w, roi['top'] + half_h
    discrim = ((x - x_cent) / half_w)**2 + ((y - y_cent) / half_h)**2
    return (discrim < 1)


ROIManager.from_zip = _from_zip
ROIManager.is_in_roi = _is_in_roi


class PlotSetting(Enum):
    NONE: Self = 'None'
    BEHAVIOR: Self = 'Behavior'
    FLUORESCENCE: Self = 'Fluorescence'
