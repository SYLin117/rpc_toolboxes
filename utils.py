import json
import glob
import os
import sys
from config import Config
from collections import defaultdict, OrderedDict
import shutil
from operator import itemgetter

cfg = Config()


def find_items():
    """
    將個編號的商品取出
    :return:
    """
    with open(
            os.path.join(cfg.get_dataset_root(), 'retail_product_checkout', 'instances_train2019.json')) as fid:
        data = json.load(fid)
    images = {}
    for x in data['images']:
        images[x['id']] = x

    annotations = {}
    for x in data['annotations']:
        annotations[images[x['image_id']]['file_name']] = x
    object_paths = glob.glob(
        os.path.join(cfg.get_dataset_root(), 'retail_product_checkout', 'train2019', '*camera3-31.jpg'))

    object_category_paths = defaultdict(list)
    for path in object_paths:
        name = os.path.basename(path)
        category = annotations[name]['category_id']
        object_category_paths[category].append(path)
    object_category_paths = dict(object_category_paths)  # store each categories all single images

    for key, values in object_category_paths.items():
        for value in values:
            code = os.path.basename(value).split('_', 1)[0]
            shutil.copyfile(
                os.path.join(cfg.get_dataset_root(), 'retail_product_checkout', 'train2019', os.path.basename(value)),
                os.path.join(cfg.get_dataset_root(), 'retail_product_checkout', 'item_images',
                             '{}-{}.jpg'.format(key, code)))


def get_sizeof_product():
    with open(
            os.path.join(cfg.get_dataset_root(), 'retail_product_checkout', 'instances_train2019.json')) as fid:
        data = json.load(fid)
    images = {}
    for x in data['images']:
        images[x['id']] = x

    annotations = {}
    for x in data['annotations']:
        annotations[images[x['image_id']]['file_name']] = x

    with open('cat_2_angle.json') as fid:
        cat_2_angle = json.load(fid)
    with open('product_code.json') as fid:
        product_code = json.load(fid)
    with open('ratio_annotations.json') as fid:
        ratio_annotations = json.load(fid)

    object_paths = list()
    rotate_angles = ['1', '12', '21', '32']
    for code, value in product_code.items():
        angles = cat_2_angle[value['sku_class']]
        if value['cat_id'] in [160, 161, 162, 163]:
            angles = [3]
        for ca in angles:
            for ra in rotate_angles:
                object_paths += glob.glob(os.path.join(cfg.get_dataset_root(), 'retail_product_checkout', 'train2019',
                                                       '{}_camera{}-{}.jpg'.format(code, ca, ra)))
                object_paths += glob.glob(os.path.join(cfg.get_dataset_root(), 'retail_product_checkout', 'train2019',
                                                       '{}-back_camera{}-{}.jpg'.format(code, ca, ra)))

    object_category_paths = defaultdict(list)
    for path in object_paths:
        name = os.path.basename(path)
        category = annotations[name]['category_id']
        object_category_paths[category].append(path)
    object_category_paths = dict(object_category_paths)  # store each categories all single images
    ratio_list = list()
    for key, values in object_category_paths.items():
        if key == 160:
            print(" ")
        min_ratio = 1
        for value in values:
            if min_ratio > ratio_annotations[os.path.basename(value)]:
                min_ratio = ratio_annotations[os.path.basename(value)]
        ratio_list.append((key, min_ratio))

    sorted(ratio_list, key=itemgetter(0))
    print(ratio_list)
    size_dict = OrderedDict()
    for id, ratio in ratio_list:

        if ratio > 0.65:
            size_dict[id] = 'large'
        elif ratio > 0.4:
            size_dict[id] = 'medium'
        else:
            size_dict[id] = 'small'
    with open("item_size.json", "w") as outfile:
        json.dump(size_dict, outfile)


if __name__ == "__main__":
    # find_items()
    get_sizeof_product()
