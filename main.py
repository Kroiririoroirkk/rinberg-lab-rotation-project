"""Calcium imaging analyzer

The purpose of this script is to allow for the convenient viewing and analysis
of two-photon calcium imaging data, together with behavioral (two-alternative
forced choice task responses) and physiological data (respiration patterns).

Author: Eric Tao (Eric.Tao@nyulangone.org)
Date created: 2025-09-23
Date last updated: 2025-10-15
"""
from typing import Self

from config import H5_PATH, H5_PICKLE_PATH, ROI_ZIP_PATH, TIFF_FOLDER
from controller import Controller
from model import Model
from view import View


class App:

    def __init__(self: Self):
        self.model = Model(TIFF_FOLDER, H5_PATH, H5_PICKLE_PATH, ROI_ZIP_PATH)
        self.view = View()
        self.controller = Controller(self.model, self.view)

    def start(self: Self):
        self.view.mainloop()


if __name__ == '__main__':
    app = App()
    app.start()
