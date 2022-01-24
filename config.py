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

        self.camera_angles = ['0', '1', '2', '3']
        self.rotate_angles = ['1', '12', '21', '32']
        # self.img_filters = ['*camera0-10*.jpg', '*camera0-30*.jpg', '*camera1-10*.jpg', ]
        self.img_filters = list()
        for ca in self.camera_angles:
            for ra in self.rotate_angles:
                self.img_filters.append('*camera{}-{}*.jpg'.format(ca, ra))

    def get_dataset_root(self):
        return self.datasets_root
