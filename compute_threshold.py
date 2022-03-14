# ===========================
# use to calculate each object(single train image) pixel ratio to the biggest one(include others)
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
with open(os.path.join(DATASET_ROOT, 'retail_product_checkout', 'instances_train2019.json')) as fid:
    data = json.load(fid)
images = {}
for x in data['images']:
    images[x['id']] = x

annotations = {}  # key: filename, value:annotation
for x in data['annotations']:
    annotations[images[x['image_id']]['file_name']] = x

# object_paths = glob.glob(os.path.join('/data7/lufficc/process_rpc/cropped_train2019/', '*.jpg'))
# object_paths = glob.glob(os.path.join(DATASET_ROOT, 'retail_product_checkout', 'crop_images', '*.jpg'))
object_paths = glob.glob(os.path.join(sys.path[0], 'extracted_masks_tracer5_morph10', 'masks', '*.png'))

object_category_paths = defaultdict(list)
for path in object_paths:
    name = os.path.basename(path).split('.')[0]
    category = annotations["{}.jpg".format(name)]['category_id']
    object_category_paths[category].append(path)  # key: category no, value: list of this category images path

object_category_paths = dict(object_category_paths)

ratio_anns = {}
all_areas = []
for category, paths in tqdm(object_category_paths.items()):
    areas = []
    for object_path in paths:
        name = os.path.basename(object_path)
        mask_path = os.path.join(sys.path[0], 'extracted_masks_tracer5_morph10', 'masks',
                                 '{}.png'.format(name.split('.')[0]))
        mask = Image.open(mask_path).convert('L')
        area = np.array(mask, dtype=bool).sum()
        areas.append(area)  # instance area
        all_areas.append(area) # new add
    # areas = np.array(areas)
    # max_area = areas.max()
    # ratios = np.round(areas / max_area, 3)
    # for i, object_path in enumerate(paths):
    #     name = os.path.basename(object_path)
    #     ratio_anns[name] = ratios[i]
all_areas = np.array(all_areas)
max_area = all_areas.max()
for category, paths in tqdm(object_category_paths.items()):
    areas = []
    for object_path in paths:
        name = os.path.basename(object_path)
        mask_path = os.path.join(sys.path[0], 'extracted_masks_tracer5_morph10', 'masks',
                                 '{}.png'.format(name.split('.')[0]))
        mask = Image.open(mask_path).convert('L')
        area = np.array(mask, dtype=bool).sum()
        ratio = np.round(area / max_area, 5) + 0.000001
        ratio_anns[name] = ratio

with open('ratio_annotations_all.json', 'w') as fid:
    json.dump(ratio_anns, fid)
