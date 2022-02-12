import json
import glob
import os
import sys
from config import Config
from collections import defaultdict, OrderedDict
import shutil
from operator import itemgetter
from pathlib import Path
import cv2
from tqdm import tqdm
import numpy as np

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
    """
    依照ratio的比例 給物件分為三種大小large, medium and small
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


def extract_val_obj():
    save_path = os.path.join(cfg.get_dataset_root(), 'retail_product_checkout', 'val_instance')
    Path.mkdir(Path(save_path).resolve(), parents=True, exist_ok=True)

    with open(os.path.join(cfg.get_dataset_root(), 'retail_product_checkout', 'instances_val2019.json'),
              'rb') as json_file:
        json_data = json.load(json_file)
    imgs = json_data['images']
    anns = json_data['annotations']
    anns = sorted(anns, key=lambda i: (i['image_id'], i['id']))
    # json中的img_id對應的檔案名稱
    image_id_2_filename = {img['id']: img['file_name'] for img in imgs}
    current_image_id = None
    img_cv = None
    ann_idx = 1
    for ann in tqdm(anns):
        if current_image_id is None:
            img_cv = cv2.imread(os.path.join(cfg.get_dataset_root(), 'retail_product_checkout', 'val2019',
                                             image_id_2_filename[ann['image_id']]))
        elif current_image_id != ann['image_id']:
            img_cv = cv2.imread(os.path.join(cfg.get_dataset_root(), 'retail_product_checkout', 'val2019',
                                             image_id_2_filename[ann['image_id']]))
        current_image_id = ann['image_id']
        bbox = ann['bbox']
        x0 = int(bbox[0])
        y0 = int(bbox[1])
        x1 = x0 + int(bbox[2])
        y1 = y0 + int(bbox[3])
        crop_img = img_cv[y0:y1, x0:x1, :]
        cv2.imwrite(os.path.join(cfg.get_dataset_root(), 'retail_product_checkout', 'val_instance',
                                 '{}-{}-{}.jpg'.format(image_id_2_filename[ann['image_id']], ann['category_id'],
                                                       ann_idx)), crop_img)
        ann_idx += 1


def check_each_class_instance_count():
    with open(os.path.join(cfg.get_dataset_root(), 'rpc_list', 'sod_synthesize_10000_0.json')) as fid:
        data = json.load(fid)
    from collections import defaultdict
    a = {}
    a = defaultdict(lambda: 0, a)
    anns = data['annotations']
    for ann in anns:
        a[ann['category_id']] += 1
    a = OrderedDict(sorted(a.items()))
    chk_idx = 1
    for k, _ in a.items():
        assert int(k) == chk_idx, "dont have category {} in dataset".format(chk_idx)
        chk_idx += 1
    print(a)


def create_synthesys_shadow(source_images_folder, source_masks_folder, new_folder):
    images = os.listdir(source_images_folder)
    masks = os.listdir(source_masks_folder)
    images.sort()
    masks.sort()

    for i, filename in enumerate(images):
        image = cv2.imread(os.path.join(source_images_folder, filename))
        mask = cv2.imread(os.path.join(source_masks_folder, filename), cv2.IMREAD_GRAYSCALE)
        shadow = np.ones_like(image.shape)
        kernel = np.ones((3, 3), np.uint8) * 10
        erosion = cv2.erode(mask, kernel, iterations=10)
        mask = cv2.dilate(erosion, kernel, iterations=10)
        shift_x = np.random.randint(1, 10) * np.random.choice([-1, 1])
        shift_y = np.random.randint(1, 10)
        M = np.float32([
            [1, 0, shift_x],
            [0, 1, shift_y]
        ])
        mask_shifted = cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]))
        diff = cv2.subtract(mask_shifted, mask)
        ret, diff = cv2.threshold(diff, 1, 1, cv2.THRESH_BINARY)
        diff = np.stack((diff, diff, diff), axis=2)

        if shift_x > shift_y:
            shift_x = shift_x / np.abs(shift_y)
            shift_y = shift_y / np.abs(shift_y)
        elif shift_x < shift_y:
            shift_y = shift_y / np.abs(shift_x)
            shift_x = shift_x / np.abs(shift_x)
        else:
            shift_x = shift_x / np.abs(shift_x)
            shift_y = shift_y / np.abs(shift_y)
        M = np.float32([
            [1, 0, shift_x * 2.5],
            [0, 1, shift_y * 2.5]
        ])
        diff = cv2.warpAffine(diff, M, (diff.shape[1], diff.shape[0]))  # diff為陰影的mask
        image = image[:, :, :] * (1 - diff * 0.7) + shadow * diff
        image = image.astype(np.uint8)
        cv2.imshow('win', image)
        # cv2.imshow('diff', mask_diff)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    print()
    # find_items()
    # get_sizeof_product()
    # extract_val_obj()
    # check_each_class_instance_count()

    create_synthesys_shadow('/Users/ianlin/datasets/rpc_list/synthesize_20000',
                            '/Users/ianlin/datasets/rpc_list/synthesize_20000_masks',
                            '/Users/ianlin/datasets/rpc_list/synthesize_20000_shadow')
