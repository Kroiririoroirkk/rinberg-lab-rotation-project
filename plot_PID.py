"""Script to plot liquid concentration vs vapor pressure graph

The purpose of this script is to quantify the deviation of an odorant chemical
such as ethyl tiglate (ET) from Raoult's law by comparing the mole fraction of
the odorant in distilled water with its vapor pressure as quantified by
photoionization (PID) measurements.

Author: Eric Tao (Eric.Tao@nyulangone.org)
Date created: 2025-10-03
Date last updated: 2025-10-03
"""
import h5py
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path('data/rotationdata/251003_passive_test2')
PID_H5 = Path(DATA_DIR, '251003_PID_0001-0120.h5')
# TODO: this is not the right file
ODOR_H5 = Path(DATA_DIR, 'mouse0953_sess26_D2025_10_2T14_21_34.h5')
FVALVE_CHANNEL = 3
PID_CHANNEL = 4


def calc_dilution():
    """Calculate the mole fraction of ET in each step of a serial dilution.

    These values are for the recording done on 2025-10-03, calculated from
    the following:
    ET1 is pure (purity ≥98%),
    ET2 is 50 μL ET1 in 5 mL distilled water,
    ET3 is 500 μL ET2 in 4.5 mL distilled water,
    ET4 is 500 μL ET3 in 4.5 mL distilled water,
    ET5 is 500 μL ET4 in 4.5 mL distilled water.

    Relevant quantities:
    ET molecular weight = 128.17 g/mol, density = 0.923 g/mL at 25 °C.
    DH2O molecular weight = 18.02 g/mol, density = 0.998 g/mL at 20 °C.
    Assuming density does not significantly change in the relevant temperature
    range.
    """
    # All molarities below written in units of mol/mL.
    ET_mw = 128.17
    ET_d = 0.923
    DH2O_mw = 18.02
    DH2O_d = 0.998
    DH2O_molarity = DH2O_d / DH2O_mw
    ET_molarity = ET_d / ET_mw

    ET2_volume = 0.05 + 5
    ET2_ET_molarity = 0.05 * ET_molarity / ET2_volume
    ET2_DH2O_molarity = 5 * DH2O_molarity / ET2_volume
    ET2_mole_fraction = ET2_ET_molarity / (ET2_ET_molarity + ET2_DH2O_molarity)

    ET3_volume = 0.5 + 4.5
    ET3_ET_molarity = 0.5 * ET2_ET_molarity / ET3_volume
    ET3_DH2O_molarity = (0.5 * ET2_DH2O_molarity +
                         4.5 * DH2O_molarity) / ET3_volume
    ET3_mole_fraction = ET3_ET_molarity / (ET3_ET_molarity + ET3_DH2O_molarity)

    ET4_volume = 0.5 + 4.5
    ET4_ET_molarity = 0.5 * ET3_ET_molarity / ET4_volume
    ET4_DH2O_molarity = (0.5 * ET3_DH2O_molarity +
                         4.5 * DH2O_molarity) / ET4_volume
    ET4_mole_fraction = ET4_ET_molarity / (ET4_ET_molarity + ET4_DH2O_molarity)

    ET5_volume = 0.5 + 4.5
    ET5_ET_molarity = 0.5 * ET4_ET_molarity / ET5_volume
    ET5_DH2O_molarity = (0.5 * ET4_DH2O_molarity +
                         4.5 * DH2O_molarity) / ET5_volume
    ET5_mole_fraction = ET5_ET_molarity / (ET5_ET_molarity + ET5_DH2O_molarity)

    return {
        b'empty': 0,
        b'ET1': 1,
        b'ET2': ET2_mole_fraction,
        b'ET3': ET3_mole_fraction,
        b'ET4': ET4_mole_fraction,
        b'ET5': ET5_mole_fraction
    }


def detect_valve_on(valve_signal):
    """Detect when the valve turns on.

    We expect that the valve graph starts at zero (<100) for at least 20000
    samples, jumps to a high state (>10000) within ten samples for at least
    20000 samples, then jumps back to zero (<100) within ten samples. If the
    valve graph does not match this shape, the function returns None.
    """
    for i in range(len(valve_signal)):
        if valve_signal[i] > 100:
            jump_on_time = i
            break
    else:
        # print('Valve never turns on.')
        return None
    if jump_on_time < 20000:
        # print('Valve needs to start off.')
        return None
    for i in range(jump_on_time + 10, len(valve_signal)):
        if valve_signal[i] < 10000:
            jump_off_time = i
            break
    else:
        # print('Valve never turns off.')
        return None
    if jump_off_time - jump_on_time < 20000:
        # print('Valve does not stay on for long enough.')
        return None
    for i in range(jump_off_time + 10, len(valve_signal)):
        if valve_signal[i] > 100:
            # print('Valve needs to stay off.')
            return None
    return jump_on_time


if __name__ == '__main__':
    pid_f = h5py.File(PID_H5)
    odor_f = h5py.File(ODOR_H5)
    # There is a key sweep_xxxx for each trial as well as a header key.
    num_trials = len(pid_f.keys()) - 1

    odors = odor_f['Trials']['olfa_1st_1_odor']
    dilution_ratios = odor_f['Trials']['olfa_1st_1_mfc_1_flow']
    mole_fractions = calc_dilution()

    mole_fractions_trial = []
    est_partial_pressure_trial = []
    for i in range(num_trials):
        mole_fraction = mole_fractions[odors[i]]
        pid_sweep = pid_f[f'sweep_{i+1:04d}']
        final_valve = pid_sweep['analogScans'][FVALVE_CHANNEL]
        pid_reading = pid_sweep['analogScans'][PID_CHANNEL]
        print(detect_valve_on(final_valve))
        plt.plot(final_valve, label='valve')
        plt.plot(pid_reading, label='PID')
        plt.legend()
        plt.show()
        # Calculate ΔV/V of PID reading, adjusted by dilution ratio
        dilution_ratio = dilution_ratios[i] / 1000
        est_partial_pressure = None  # TODO
        mole_fractions_trial.append(mole_fraction)
        est_partial_pressure_trial.append(est_partial_pressure)
    plt.scatter(mole_fractions_trial, est_partial_pressure)
    plt.title('Raoult\'s law plot for ethyl tiglate')
    plt.xlabel('Mole fraction of ET in liquid mixture')
    plt.ylabel('Partial pressure (relative units)')
    plt.show()
