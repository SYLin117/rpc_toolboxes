# ===========================
# use to calculate each object(single train image) pixel ratio to the biggest one to it's own kind
# ===========================
import glob
import json
import os
from collections import defaultdict

import numpy as np
from PIL import Image
from tqdm import tqdm
from config import Config
import sys

cfg = Config()
DATASET_ROOT = cfg.get_dataset_root()


def compute():
    with open(os.path.join(DATASET_ROOT, 'retail_product_checkout', 'instances_train2019.json')) as fid:
        data = json.load(fid)
    images = {}
    for x in data['images']:
        images[x['id']] = x

    annotations = {}
    for x in data['annotations']:
        annotations[images[x['image_id']]['file_name']] = x

    # object_paths = glob.glob(os.path.join('/data7/lufficc/process_rpc/cropped_train2019/', '*.jpg'))
    # object_paths = glob.glob(os.path.join('/media/ian/WD/PythonProject/DP-Net/acm-mm-2019-ACO-master/toolboxes/extracted_masks/crop_images/', '*.jpg'))
    object_paths = glob.glob(os.path.join(DATASET_ROOT, 'retail_product_checkout', 'train2019', '*.jpg'))

    object_category_paths = defaultdict(list)
    for path in object_paths:
        name = os.path.basename(path)
        category = annotations[name]['category_id']
        object_category_paths[category].append(path)  # key: category no, value: list of this category images path

    object_category_paths = dict(object_category_paths)

    ratio_anns = {}
    for category, paths in tqdm(object_category_paths.items()):
        areas = []
        for object_path in paths:
            name = os.path.basename(object_path)
            mask_path = os.path.join(sys.path[0], 'extracted_masks_tracer5_morph10', 'masks',
                                     '{}.png'.format(name.split('.')[0]))
            mask = Image.open(mask_path).convert('L')
            area = np.array(mask, dtype=np.bool).sum()
            areas.append(area)  # instance area
        areas = np.array(areas)
        max_area = areas.max()
        ratios = np.round(areas / max_area, 3)
        for i, object_path in enumerate(paths):
            name = os.path.basename(object_path)
            ratio_anns[name] = ratios[i]
    with open('ratio_annotations.json', 'w') as fid:
        json.dump(ratio_anns, fid)


if __name__ == '__main__':
    compute()
