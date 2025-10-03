"""Toy script to play with reading PID data from H5 file

Only for practice, not for research.

Author: Eric Tao (Eric.Tao@nyulangone.org)
Date created: 2025-10-03
Date last updated: 2025-10-03
"""
import h5py
import matplotlib.pyplot as plt
from pathlib import Path

H5_FILE = Path('data/rotationdata/PID_test_0001-0014.h5')
N = 14

if __name__ == '__main__':
    f = h5py.File(H5_FILE)
    for i in range(N):
        fig, axes = plt.subplots(2, sharex=True)
        s = f[f'sweep_{i+1:04d}']
        final_valve = s['analogScans'][3]
        pid_reading = s['analogScans'][4]
        axes[0].plot(final_valve)
        axes[1].plot(pid_reading)
        fig.suptitle(f'Trial {i+1}, valve and exhaust PID readings')
        plt.show()
