from sys import platform
from pathlib import Path
import os
import json


class Config:
    def __init__(self):
        self.datasets_root = None
        if platform == "linux" or platform == "linux2":  # on linux machine
            # linux
            self.datasets_root = str(Path(r'/media/ian/WD/datasets'))
        elif platform == "darwin":  # on mac
            # OS X
            self.datasets_root = str(Path(r'/media/ian/WD/datasets'))
        elif platform == "win32":  # on Windows
            # Windows...
            self.datasets_root = str(Path(r'D:\datasets'))

        self.camera_angles = ['1', '3']
        self.rotate_angles = ['3', '18', '28']
        # self.img_filters = ['*camera0-10*.jpg', '*camera0-30*.jpg', '*camera1-10*.jpg', ]
        self.img_filters = list()
        for ca in self.camera_angles:
            for ra in self.rotate_angles:
                self.img_filters.append('*camera{}-{}*.jpg'.format(ca, ra))

    def get_dataset_root(self):
        return self.datasets_root

    def get_dataset_info(self):
        json_info = None
        with open(os.path.join('instances_train2019.json')) as fid:
            json_info = json.load(fid)
        return json_info
