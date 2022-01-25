import cv2
from config import Config
import os
import numpy as np
import glob

cfg = Config()
DATASET_ROOT = cfg.get_dataset_root()

splices = glob.glob(os.path.join(DATASET_ROOT, 'license_plate', 'images', 'IM_*-S*.jpg'))
copys = glob.glob(os.path.join(DATASET_ROOT, 'license_plate', 'images', 'IM_*-C*.jpg'))

for img in splices:
    filename = os.path.basename(img)
    img1_path = os.path.join(DATASET_ROOT, 'license_plate', 'images', '{}.jpg'.format(filename.split('-')[0]))
    img2_path = os.path.join(DATASET_ROOT, 'license_plate', 'images', filename)

    img1 = cv2.imread(img1_path, cv2.IMREAD_COLOR)
    img2 = cv2.imread(img2_path, cv2.IMREAD_COLOR)

    diff = cv2.absdiff(img1, img2)
    mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(mask, 15, 255, cv2.THRESH_BINARY)

    th = 10
    imask = mask > th

    canvas = np.zeros_like(img2, np.uint8)
    canvas[imask] = img2[imask]

    cv2.imwrite(os.path.join(DATASET_ROOT, 'license_plate', 'masks', '{}.png'.format(filename.split('.')[0])), thresh)
