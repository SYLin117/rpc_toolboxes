import json
from config import Config
import os
import glob
from collections import defaultdict
from shutil import copyfile

config = Config()
DATASET_ROOT = config.get_dataset_root()
with open(os.path.join(DATASET_ROOT, 'retail_product_checkout', 'instances_train2019.json')) as fid:
    data = json.load(fid)
images = {}
for x in data['images']:
    images[x['id']] = x

annotations = {}
for x in data['annotations']:
    annotations[images[x['image_id']]['file_name']] = x

object_paths = glob.glob(
    os.path.join(DATASET_ROOT, 'retail_product_checkout', 'train2019', '*camera1-10*.jpg'))

object_category_paths = defaultdict(list)
for path in object_paths:
    name = os.path.basename(path)
    category = annotations[name]['category_id']
    object_category_paths[category].append(path)
object_category_paths = dict(object_category_paths)  # store each categories all single images
print('================================================')
save_sub_dir = os.path.join(DATASET_ROOT, 'retail_product_checkout', 'items')
if not os.path.exists(save_sub_dir):
    os.mkdir(save_sub_dir)
for key, values in object_category_paths.items():
    class_name = next(item for item in data['categories'] if item["id"] == key)
    class_folder = os.path.join(save_sub_dir, class_name['name'])
    if not os.path.exists(class_folder):
        os.mkdir(class_folder)
    for value in values:
        copyfile(value, os.path.join(class_folder, os.path.basename(value)))
