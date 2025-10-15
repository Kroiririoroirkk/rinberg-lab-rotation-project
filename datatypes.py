from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import NewType, Self

import h5py
import numpy as np
import read_roi
from numpy.typing import NDArray

from config import ODOR_DICT

Odor = NewType('Odor', str)


def _odor_lookup(s: bytes) -> Odor | None:
    return ODOR_DICT.get(s)


Odor.lookup = _odor_lookup
Odor.EMPTY = Odor('Empty')


class Choice(Enum):
    LEFT: Self = 'Left'
    RIGHT: Self = 'Right'
    MISS: Self = 'Miss'
    PASSIVE: Self = 'Passive'

    @staticmethod
    def lookup_target(i: int) -> Self | None:
        d = {0: Choice.LEFT, 1: Choice.RIGHT, 4: Choice.PASSIVE}
        return d.get(i)

    @staticmethod
    def lookup_response(i: int) -> Self | None:
        d = {
            0: Choice.PASSIVE,
            1: Choice.RIGHT,
            2: Choice.LEFT,
            3: Choice.LEFT,
            4: Choice.RIGHT,
            5: Choice.MISS,
            6: Choice.MISS
        }
        return d.get(i)


def _read_clumped_arr(arr: h5py._hl.dataset.Dataset) -> NDArray[np.int64]:
    return np.array([int(x) for x in np.concatenate(arr)],
                    dtype=np.int64) if arr.size else np.array([])


StimID = NewType('StimID', int)


@dataclass(frozen=True)
class StimCondition:
    odor1: Odor
    odor2: Odor
    odor1_flow: float
    odor2_flow: float


@dataclass(frozen=True)
class TrialMetadata:
    frame_times: NDArray[np.int64]  # in ms
    odor_time: int  # in ms
    lick1_times: NDArray[np.int64]  # in ms
    lick2_times: NDArray[np.int64]  # in ms
    sniff_times: NDArray[np.int64]  # in ms
    sniff_values: NDArray[np.int64]  # in relative units
    stim_condition: StimCondition
    target: Choice
    response: Choice
    correct: bool
    stim_id: StimID

    @staticmethod
    def list_from_h5(h5_path: Path) -> list[Self | str]:
        trial_metadata_list = []
        stim_condition_dict = dict()
        f = h5py.File(h5_path, 'r')
        all_trials_f = f['Trials']
        recorded_trials = np.nonzero(all_trials_f['record'])[0]
        for i in recorded_trials:
            try:
                trial_f = f[f'Trial{i+1:04d}']
                frame_times = _read_clumped_arr(trial_f['frame_triggers'])
                if frame_times.size == 0:
                    # This trial was not saved as a TIFF file
                    continue
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
                        f'Sniff chunk lengths not consistent for '
                        f'Trial{i+1:04d}: '
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
                sniff_times = np.array(sniff_times, dtype=np.int64)
                sniff_values = _read_clumped_arr(trial_f['sniff'][:-1])
                if sniff_times.size != sniff_values.size:
                    raise ValueError(
                        'Sniff time and sniff value lists do not match up.')
                o1, o2 = all_trials_f['olfa_1st_0_odor'][i], all_trials_f[
                    'olfa_1st_1_odor'][i]
                odor1, odor2 = Odor.lookup(o1), Odor.lookup(o2)
                if odor1 is None or odor2 is None:
                    raise ValueError(
                        f'Odor lookup failed on odors: {o1}, {o2}.')
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
                stim_id = StimID(int(all_trials_f['stimid'][i]))
                stim_condition = StimCondition(odor1=odor1,
                                               odor2=odor2,
                                               odor1_flow=odor1_flow,
                                               odor2_flow=odor2_flow)
                if stim_id in stim_condition_dict:
                    if stim_condition != stim_condition_dict[stim_id]:
                        raise ValueError(
                            f'Stim ID {stim_id} has multiple stim condition '
                            f'values (failed on input: {r}).')
                else:
                    stim_condition_dict[stim_id] = stim_condition
                trial_metadata_list.append(
                    TrialMetadata(frame_times=frame_times,
                                  odor_time=odor_time,
                                  lick1_times=lick1_times,
                                  lick2_times=lick2_times,
                                  sniff_times=sniff_times,
                                  sniff_values=sniff_values,
                                  stim_condition=stim_condition,
                                  target=target,
                                  response=response,
                                  correct=correct,
                                  stim_id=stim_id))
            except ValueError as e:
                print(i)
                trial_metadata_list.append(str(e))
        return trial_metadata_list, stim_condition_dict


ROIName = NewType('ROIName', str)
ROI = NewType('ROI', dict[str, str | int])
ROIManager = NewType('ROIManager', OrderedDict[ROIName, ROI])


def _from_zip(roi_zip_path: Path) -> ROIManager:
    rois = read_roi.read_roi_zip(roi_zip_path)
    for v in rois.values():
        if v.get('type') != 'oval':
            raise ValueError('Cannot process non-oval ROI.')
    roi_names, rois = list(zip(*rois.items()))
    position_arr = [(roi['top'] + roi['height'] / 2,
                     roi['left'] + roi['width'] / 2, i)
                    for i, roi in enumerate(rois)]
    position_arr.sort()
    idx = [pos[-1] for pos in position_arr]
    return ROIManager({ROIName(roi_names[i]): ROI(rois[i]) for i in idx})


def _is_in_roi(x: int, y: int, roi: ROI) -> bool:
    half_w, half_h = roi['width'] / 2, roi['height'] / 2
    x_cent, y_cent = roi['left'] + half_w, roi['top'] + half_h
    discrim = ((x - x_cent) / half_w)**2 + ((y - y_cent) / half_h)**2
    return (discrim < 1)


ROIManager.from_zip = _from_zip
ROIManager.is_in_roi = _is_in_roi


class PageSetting(Enum):
    BY_TRIAL: str = 'By trial'
    BY_STIM_ID: str = 'By stim ID'


def for_page(d: dict[PageSetting, type], s: PageSetting):

    def decorator(cls: type):
        d[s] = cls
        return cls

    return decorator


class ByTrialPlotSetting(Enum):
    NONE: Self = 'None'
    BEHAVIOR: Self = 'Behavior'
    FLUORESCENCE: Self = 'Fluorescence'


class ByStimIDPlotSetting(Enum):
    NONE: Self = 'None'
    FLUORESCENCE: Self = 'Fluorescence'
    ODOR_1_RESPONSE: Self = 'Odor 1 response'
    ODOR_2_RESPONSE: Self = 'Odor 2 response'
    ODOR_1_LATENCY: Self = 'Odor 1 latency'
    ODOR_2_LATENCY: Self = 'Odor 2 latency'
    GLOM_MAX_RESPONSE: Self = 'Glom max response'
    NUM_GLOMS_ACTIVE: Self = 'Glom count'
