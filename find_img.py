import json

from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import cv2

import os
import numpy as np
import random
from config import Config

"""
將自己標記的ann轉換成mask作為SOD模型的training資料
"""
dataType = "test2019"
config = Config()
DATASET_ROOT = config.get_dataset_root()
RPC_ROOT = os.path.join(DATASET_ROOT, 'retail_product_checkout')
annFile = os.path.join(RPC_ROOT, f'instances_{dataType}.json')

with open(annFile) as file:
    data = json.load(file)

images = {}
for x in data['images']:
    images[x['id']] = x

annotations = {}
for x in data['annotations']:
    if images[x['image_id']]['file_name'] in annotations:
        annotations[images[x['image_id']]['file_name']].append(x)
    else:
        annotations[images[x['image_id']]['file_name']] = list()
        annotations[images[x['image_id']]['file_name']].append(x)

print('==========================')

hardness = 'easy'
contained = [72]

match_hardness = dict()
for key, value in images.items():
    if value['level'] == hardness:
        match_hardness[value['file_name']] = value

match_contained = dict()
for key, value in annotations.items():
    for obj in value:
        if obj['category_id'] in contained:
            match_contained[key] = value

print('==========================')

for (key) in set(match_hardness.keys()) & set(match_contained.keys()):
    print('%s: is present in both match_hardness and match_contained' % (key))
