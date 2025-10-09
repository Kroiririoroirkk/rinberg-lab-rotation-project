from typing import Self

from config import (H5_PATH, H5_PICKLE_PATH, ROI_ZIP_PATH, TIFF_FOLDER)
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
