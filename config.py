from sys import platform
from pathlib import Path


class Config:
    def __init__(self):
        self.datasets_root = None
        if platform == "linux" or platform == "linux2":  # on linux machine
            # linux
            self.datasets_root = str(Path(r'/media/ian/WD/datasets'))
        elif platform == "darwin":  # on mac
            # OS X
            raise Exception("no dataset on mac.")
        elif platform == "win32":  # on Windows
            # Windows...
            self.datasets_root = str(Path(r'D:\datasets'))

        self.img_filter = ['*camera1-10*.jpg']

    def get_dataset_root(self):
        return self.datasets_root
