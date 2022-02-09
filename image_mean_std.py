import numpy as np
import cv2
import os
import re
from tqdm import tqdm


def calc_avg_mean_std(img_names, img_root, size):
    mean_sum = np.array([0., 0., 0.])
    std_sum = np.array([0., 0., 0.])
    n_images = len(img_names)
    for img_name in tqdm(img_names):
        img = cv2.imread(os.path.join(img_root, img_name))
        img = cv2.resize(img, size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mean, std = cv2.meanStdDev(img)
        mean_sum += np.squeeze(mean)
        std_sum += np.squeeze(std)
    return (mean_sum / n_images, std_sum / n_images)


train_img_root = './synthesize_20000'
train_img_names = [f for f in os.listdir(train_img_root) if re.match(r'[a-zA-Z0-9]+.*\.jpg', f)]
train_mean, train_std = calc_avg_mean_std(train_img_names, train_img_root, (1815, 1815))
print(train_mean)
print(train_std)
