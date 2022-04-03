import json
import glob
import os
import random
import sys

import pandas as pd

from config import Config
from collections import defaultdict, OrderedDict
import shutil
from operator import itemgetter
from pathlib import Path
import cv2
from tqdm import tqdm
import numpy as np
from pycocotools.coco import COCO
# from chitra.image import Chitra
import matplotlib.pyplot as plt
import time
import skimage
from typing import Optional
import seaborn as sns

import torch
import torchvision.transforms as T
import tracer

from PIL import Image

RPC_CLASSES = (
    '1_puffed_food', '2_puffed_food', '3_puffed_food', '4_puffed_food', '5_puffed_food',
    '6_puffed_food', '7_puffed_food',
    '8_puffed_food', '9_puffed_food', '10_puffed_food', '11_puffed_food', '12_puffed_food', '13_dried_fruit',
    '14_dried_fruit', '15_dried_fruit',
    '16_dried_fruit', '17_dried_fruit', '18_dried_fruit', '19_dried_fruit', '20_dried_fruit',
    '21_dried_fruit', '22_dried_food', '23_dried_food',
    '24_dried_food', '25_dried_food', '26_dried_food', '27_dried_food', '28_dried_food', '29_dried_food',
    '30_dried_food', '31_instant_drink',
    '32_instant_drink', '33_instant_drink', '34_instant_drink', '35_instant_drink', '36_instant_drink',
    '37_instant_drink', '38_instant_drink',
    '39_instant_drink', '40_instant_drink', '41_instant_drink', '42_instant_noodles', '43_instant_noodles',
    '44_instant_noodles',
    '45_instant_noodles', '46_instant_noodles', '47_instant_noodles', '48_instant_noodles',
    '49_instant_noodles', '50_instant_noodles',
    '51_instant_noodles', '52_instant_noodles', '53_instant_noodles', '54_dessert', '55_dessert',
    '56_dessert', '57_dessert', '58_dessert',
    '59_dessert', '60_dessert', '61_dessert', '62_dessert', '63_dessert', '64_dessert', '65_dessert',
    '66_dessert', '67_dessert', '68_dessert',
    '69_dessert', '70_dessert', '71_drink', '72_drink', '73_drink', '74_drink', '75_drink', '76_drink',
    '77_drink', '78_drink', '79_alcohol',
    '80_alcohol', '81_drink', '82_drink', '83_drink', '84_drink', '85_drink', '86_drink', '87_drink',
    '88_alcohol', '89_alcohol', '90_alcohol',
    '91_alcohol', '92_alcohol', '93_alcohol', '94_alcohol', '95_alcohol', '96_alcohol', '97_milk', '98_milk',
    '99_milk', '100_milk', '101_milk',
    '102_milk', '103_milk', '104_milk', '105_milk', '106_milk', '107_milk', '108_canned_food',
    '109_canned_food', '110_canned_food',
    '111_canned_food', '112_canned_food', '113_canned_food', '114_canned_food', '115_canned_food',
    '116_canned_food', '117_canned_food',
    '118_canned_food', '119_canned_food', '120_canned_food', '121_canned_food', '122_chocolate',
    '123_chocolate', '124_chocolate', '125_chocolate',
    '126_chocolate', '127_chocolate', '128_chocolate', '129_chocolate', '130_chocolate', '131_chocolate',
    '132_chocolate', '133_chocolate', '134_gum',
    '135_gum', '136_gum', '137_gum', '138_gum', '139_gum', '140_gum', '141_gum', '142_candy', '143_candy',
    '144_candy', '145_candy', '146_candy',
    '147_candy', '148_candy', '149_candy', '150_candy', '151_candy', '152_seasoner', '153_seasoner',
    '154_seasoner', '155_seasoner', '156_seasoner',
    '157_seasoner', '158_seasoner', '159_seasoner', '160_seasoner', '161_seasoner', '162_seasoner',
    '163_seasoner', '164_personal_hygiene',
    '165_personal_hygiene', '166_personal_hygiene', '167_personal_hygiene', '168_personal_hygiene',
    '169_personal_hygiene', '170_personal_hygiene',
    '171_personal_hygiene', '172_personal_hygiene', '173_personal_hygiene', '174_tissue', '175_tissue',
    '176_tissue', '177_tissue', '178_tissue',
    '179_tissue', '180_tissue', '181_tissue', '182_tissue', '183_tissue', '184_tissue', '185_tissue',
    '186_tissue', '187_tissue', '188_tissue',
    '189_tissue', '190_tissue', '191_tissue', '192_tissue', '193_tissue', '194_stationery', '195_stationery',
    '196_stationery', '197_stationery',
    '198_stationery', '199_stationery', '200_stationery'
)

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
    with open('ratio_annotations_all.json') as fid:
        ratio_annotations = json.load(fid)
    class_2_product_code = {}
    for k, v in product_code.items():
        class_2_product_code["{}_{}".format(v['cat_id'], v['sku_class'])] = k
    train_imgs_dir = os.path.join(cfg.get_dataset_root(), 'retail_product_checkout', 'train2019')  # rpc 的train影像資料夾
    object_paths = []
    rotate_angles = [str(x) for x in range(1, 40, 3)]
    for code, value in product_code.items():
        angles = cat_2_angle[value['sku_class']]
        if value['cat_id'] in [160, 161, 162, 163]:  # 瓶裝的調味料(seasoner)(其他是包裝的)
            angles = [3]
        elif value['cat_id'] in [37, 39, 40, 41]:  # 包裝類的即溶飲料(instant-drink)(其他是罐裝的)
            angles = [1]
        elif value['cat_id'] in [134, 135, 144]:  # canned candy(not package)
            angles = [0]
        elif value['cat_id'] in [45, 46, 47, 48, 49]:  # cup noodle
            angles = [0, 2]
        elif value['cat_id'] in [198, 200]:  # cylinder-like
            angles = [0]
        for ca in angles:  # camera angle
            for ra in rotate_angles:  # rotate angle
                object_paths += glob.glob(os.path.join(train_imgs_dir, '{}_camera{}-{}.jpg'.format(code, ca, ra)))
                object_paths += glob.glob(
                    os.path.join(train_imgs_dir, '{}-back_camera{}-{}.jpg'.format(code, ca, ra)))  # 有背面的商品
    tissue_remove_paths = []
    tissue_nos = [i for i in range(174, 194)]
    tissue_remove_rotate_angles = [i for i in range(3, 17)] + [i for i in range(22, 38)]
    for tissue_no in tissue_nos:
        code = class_2_product_code["{}_tissue".format(tissue_no)]
        angles = [0, 1, 2, 3]
        for ca in angles:  # camera angle
            for ra in tissue_remove_rotate_angles:
                tissue_remove_paths += glob.glob(
                    os.path.join(train_imgs_dir, '{}_camera{}-{}.jpg'.format(code, ca, ra)))
                tissue_remove_paths += glob.glob(
                    os.path.join(train_imgs_dir, '{}-back_camera{}-{}.jpg'.format(code, ca, ra)))
    object_paths = list(set(object_paths).difference(set(tissue_remove_paths)))

    object_category_paths = defaultdict(list)
    for path in object_paths:
        name = os.path.basename(path)
        category = annotations[name]['category_id']
        object_category_paths[category].append(path)
    object_category_paths = dict(object_category_paths)  # store each categories all single images
    ratio_list = list()
    for key, values in object_category_paths.items():  # key: product_id, values:file paths
        ratios = []
        for value in values:
            ratios.append(ratio_annotations["{}.png".format(os.path.basename(value).split('.')[0])])
        ratios = np.array(ratios)
        ratio_list.append((key, ratios.max()))

    sorted(ratio_list, key=itemgetter(0))
    print(ratio_list)
    size_dict = OrderedDict()
    for id, ratio in ratio_list:
        if ratio > 0.6:
            size_dict[id] = 'super_large'
        if ratio > 0.4:
            size_dict[id] = 'extra_large'
        elif ratio > 0.2:
            size_dict[id] = 'large'
        elif ratio > 0.166:
            size_dict[id] = 'medium'
        elif ratio > 0.133:
            size_dict[id] = 'small'
        elif ratio > .08:
            size_dict[id] = 'little'
        elif ratio <= 0.04:
            size_dict[id] = 'tiny'
        else:
            size_dict[id] = 'little'
    order_size_dict = OrderedDict(sorted(size_dict.items()))
    with open("item_size_all.json", "w") as outfile:
        json.dump(order_size_dict, outfile, indent=4)


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


def create_synthesys_shadow2(source_images_folder, source_masks_folder, new_folder):
    images = os.listdir(source_images_folder)
    masks = os.listdir(source_masks_folder)
    images.sort()
    masks.sort()
    for i, filename in enumerate(images):
        image = cv2.imread(os.path.join(source_images_folder, filename), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(os.path.join(source_masks_folder, filename), cv2.IMREAD_GRAYSCALE)
        edges = cv2.Canny(image, 100, 200)
        cv2.imshow('win', edges)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def resize_images(files: list, target_size: int, interpolation: int, target_folder: str = None, isMask: bool = False):
    """
    Args:
        files: list of images path
        target_size: int
        interpolation: cv2.INTER_...

    Returns:
    """
    if target_folder:
        os.makedirs(target_folder, exist_ok=True)
    if not isMask:
        for file in tqdm(files):
            filename = os.path.basename(file)
            img_cv = cv2.imread(file, cv2.IMREAD_COLOR)
            img_cv = cv2.resize(img_cv, (target_size, target_size), interpolation=interpolation)
            if not target_folder:
                cv2.imwrite(file, img_cv)
            else:
                cv2.imwrite(os.path.join(target_folder, filename), img_cv)
    else:
        for file in tqdm(files):
            filename = os.path.basename(file)
            img_cv = cv2.imread(file, cv2.IMREAD_COLOR)
            img_cv = cv2.resize(img_cv, (target_size, target_size), interpolation=interpolation)
            # kernel =
            # img_cv = skimage.transform.resize(img_cv,
            #                                   (target_size, target_size),
            #                                   mode='edge',
            #                                   anti_aliasing=False,
            #                                   anti_aliasing_sigma=None,
            #                                   preserve_range=True,
            #                                   order=0)
            if not target_folder:
                cv2.imwrite(file, img_cv)
            else:
                cv2.imwrite(os.path.join(target_folder, filename), img_cv)


def rescale_coco_data(image_folder: str, mask_folder: Optional[str],
                      json_file: str,
                      target_size: int,
                      target_image_folder: str,
                      target_mask_folder: Optional[str], target_json_file: str):
    """
    rescale images, masks and bounding box in json file of coco style data
    after small the dataset
    could use rpc_mask_2_coco.py convert mask to seg datas
    Args:
        image_folder: path of images folder
        mask_folder:  path of masks folder
        json_file: path of json file
    Returns: None
    """
    scale = target_size / 1815
    our_coco = COCO(json_file)
    images = glob.glob(os.path.join(image_folder, "*.jpg"))
    if mask_folder:
        masks = glob.glob(os.path.join(mask_folder, "*.png"))
    json_data = None
    with open(json_file) as fid:
        json_data = json.load(fid)
    new_anns = []
    ann_idx = 0
    imgToAnns = our_coco.imgToAnns
    for k, v in imgToAnns.items():  # k: img_id, v: anns
        img_path = os.path.join(image_folder, our_coco.imgs[k]['file_name'])
        for i, ann in enumerate(v):
            new_anns.append({
                'bbox': (
                    ann['bbox'][0] * scale, ann['bbox'][1] * scale, ann['bbox'][2] * scale, ann['bbox'][3] * scale),
                'category_id': ann['category_id'],
                'image_id': k,
                'iscrowd': 0,
                'id': ann_idx
            })
            ann_idx += 1
    print("... resize images ...")
    resize_images(images, target_size, cv2.INTER_CUBIC, target_image_folder)
    print("... resize mask ...")
    if mask_folder is not None:
        resize_images(masks, target_size, cv2.INTER_NEAREST_EXACT, target_mask_folder, isMask=True)
    json_images = json_data['images']
    for img in json_images:
        img['width'] = target_size
        img['height'] = target_size
    json_data['images'] = json_images
    json_data['annotations'] = new_anns
    with open(target_json_file, 'w') as fid:
        json.dump(json_data, fid)
    # chitra_image = Chitra(img_path, bboxes=bboxes, labels=labels)
    # img, bbox = chitra_image.resize_image_with_bbox([500, 500])
    # plt.imshow(chitra_image.draw_boxes())
    # plt.show()


def check_ann_duplicate_id(json_file: str):
    """
    check if there are duplicate annotation id in json file
    Args:
        json_file: str path

    Returns: None

    """
    with open(json_file) as fid:
        data = json.load(fid)
    id_list = []
    annotations = data['annotations']
    for ann in annotations:
        id_list.append(ann['id'])
    print(len(id_list) != len(set(id_list)))


def check_coco_seg_format(json_path):
    """
    check there is invalid segmentation data in coco json file and remove invalid seg then resave it
    Args:
        json_path: path to json file

    Returns: None
    """
    with open(json_path) as fid:
        json_data = json.load(fid)
    annotations = json_data['annotations']
    for annotation in tqdm(annotations):
        segs = annotation['segmentation']
        new_segs = []
        for seg in segs:
            len_seg = len(seg)
            if len_seg % 2 == 0 and len_seg >= 4:
                new_segs.append(seg)
            if len_seg % 2 != 0:
                print('get an annotation not even.')
            if len_seg <= 4:
                print('seg len less than 4: {}'.format(len_seg))
                print('annoation: {}'.format(annotation))
        annotation['segmentation'] = new_segs
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)


def visualize_bbox(json_path, img_path, save_path, COLORS: np.ndarray, CLASSES: tuple, rescale: int = None):
    """

    Args:
        json_path: path of json file
        img_path:  images folder path
        save_path: new folder to save labeled images
        COLORS:
        CLASSES:
        rescale:

    Returns:

    """

    def vis(img, box, cls_id, class_names=None):
        cls_id = cls_id - 1

        x0 = int(box[0])
        y0 = int(box[1])
        x1 = x0 + int(box[2])
        y1 = y0 + int(box[3])

        color = (COLORS[cls_id] * 255).astype(np.uint8).tolist()
        text = '{}'.format(class_names[cls_id], )
        txt_color = (0, 0, 0) if np.mean(COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.5, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        txt_bk_color = (COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.5, txt_color, thickness=1)

        return img

    with open(json_path, 'rb') as json_file:
        json_data = json.load(json_file)
    os.makedirs(save_path, exist_ok=True)
    imgs = json_data['images']
    anns = json_data['annotations']
    anns = sorted(anns, key=lambda i: (i['image_id'], i['id']))
    image_id_2_filename = {img['id']: img['file_name'] for img in imgs}
    current_image_id = None
    img_cv = None
    # idx = 0
    for ann in tqdm(anns):
        if current_image_id != ann['image_id']:
            if current_image_id is not None:
                cv2.imwrite(os.path.join(save_path, image_id_2_filename[current_image_id]), img_cv)
            current_image_id = ann['image_id']
            img_cv = cv2.imread(os.path.join(img_path, image_id_2_filename[current_image_id]))
            vis(img_cv, ann['bbox'], ann['category_id'], CLASSES)
        else:
            vis(img_cv, ann['bbox'], ann['category_id'], CLASSES)


def check_ratio(json_path):
    """
    檢查個圖片的ratio
    :param json_path:
    :return:
    """
    with open(json_path) as fid:
        ratio_dict = json.load(fid)
    for k, v in ratio_dict.items():
        ratio_dict[k] = round(ratio_dict[k], 2)
    ratio_list = [val for _, val in ratio_dict.items()]
    data = {'values': ratio_list}
    df = pd.DataFrame.from_dict(data)
    # plt.hist(ratio_list, color='g', bins=.05)
    sns.histplot(df, binwidth=.02)
    # plt.bar(list(dictionary.keys()), dictionary.values(), color='g')
    plt.show()


def add_area_2_bbox(json_path):
    with open(json_path) as fid:
        json_data = json.load(fid)
    annotations = json_data['annotations']
    for ann in annotations:
        ann['area'] = round(ann['bbox'][2] * ann['bbox'][3])
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f)


def extract_val_ann(json_path, image_root=None, save_root=None, mask_save_root=None):
    """
    將
    :param json_path:
    :param image_root:
    :return:
    """
    os.makedirs(save_root, exist_ok=True)
    with open(json_path) as fid:
        val_data = json.load(fid)
    val_coco = COCO(annotation_file=json_path)
    annotations = val_data['annotations']
    img_dict = val_coco.imgs
    img_with_ann_dict = val_coco.imgToAnns
    for img_id, anns in img_with_ann_dict.items():
        img_info = img_dict[img_id]
        img_cv = cv2.imread(os.path.join(image_root, img_info['file_name']))
        for ann in anns:
            x, y, w, h = ann['bbox']
            x = int(x)
            y = int(y)
            w = int(w)
            h = int(h)

            ann_area = img_cv[y:y + h, x:x + w]
            if save_root:
                cv2.imwrite(os.path.join(save_root, "{}.jpg".format(ann['id'])), ann_area)


def create_syn_val(json_path, image_root=None, ann_root=None, mask_root=None, save_root=None, ):
    """
    create synthesis image of val2019
    :param json_path:
    :param image_root:
    :return:
    """
    if save_root:
        os.makedirs(save_root, exist_ok=True)
    with open(json_path) as fid:
        val_data = json.load(fid)
    val_coco = COCO(annotation_file=json_path)
    img_dict = val_coco.imgs
    img_with_ann_dict = val_coco.imgToAnns
    for img_id, anns in tqdm(img_with_ann_dict.items()):
        img_info = img_dict[img_id]
        img = Image.open(os.path.join(image_root, img_info['file_name']))
        img_w, img_h = img.size
        bg_img = Image.open(os.path.join(sys.path[0], 'bg1.jpg'))
        bg_img = bg_img.resize((img_w, img_h))
        for ann in anns:
            x, y, w, h = ann['bbox']
            x = int(x)
            y = int(y)
            w = int(w)
            h = int(h)
            ann_img = Image.open(os.path.join(ann_root, '{}.jpg'.format(ann['id'])))
            mask = Image.open(os.path.join(mask_root, '{}.png'.format(ann['id']))).convert('1')
            bg_img.paste(ann_img, box=(x, y), mask=mask)
        # plt.imshow(bg_img)
        # plt.title(img_info['file_name'])
        # plt.show()
        bg_img.save(os.path.join(save_root, img_info['file_name']))
        # time.sleep(5)


def extract_ann_mask(image_root, save_root, compare_root):
    if save_root:
        os.makedirs(save_root, exist_ok=True)
    if compare_root:
        os.makedirs(compare_root, exist_ok=True)

    def normPRED_np(d):
        ma = np.max(d)
        mi = np.min(d)

        dn = (d - mi) / (ma - mi)
        return dn

    ## ========= setup model ================
    tracer_cfg = tracer.getConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dir = './weights/tracer5_best.pth'
    net = tracer.TRACER(tracer_cfg)
    net.load_state_dict(torch.load(model_dir))
    net = net.to(device)
    net.eval()
    tf = T.Compose([
        T.ToPILImage(),
        T.Resize([tracer_cfg.img_size, tracer_cfg.img_size]),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    ## ======================================
    paths = glob.glob(os.path.join(image_root, '*.jpg'))
    for img_path in paths:
        img_cv = cv2.imread(img_path)
        h, w, _ = img_cv.shape
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        img_tf = tf(img_rgb)
        img_tf = img_tf.unsqueeze(0)
        img_tf = torch.tensor(img_tf, device=device, dtype=torch.float32)
        pred_tf, _, _ = net(img_tf)
        ## ------------------------
        pred_np = pred_tf.cpu().detach().numpy()
        pred = pred_np[:, 0, :, :]
        pred = normPRED_np(pred)
        pred = pred.squeeze()
        pred = pred * 255
        pred.astype(np.uint8)
        ret, thresh = cv2.threshold(pred, 125, 255, cv2.THRESH_BINARY)
        thresh = np.array(thresh, np.uint8)
        ## ------------------------
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        threshed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=5)
        thresh_stack = np.stack((threshed,) * 3, axis=-1)  # trans to 3 channel
        thresh_stack = cv2.resize(thresh_stack, (w, h), interpolation=cv2.INTER_NEAREST)
        filled = cv2.cvtColor(thresh_stack, cv2.COLOR_BGR2GRAY)
        filled = filled / 255
        ##--------------------------
        save_mask = np.array(filled * 255, dtype=np.uint8)
        masked_img = img_cv * filled[:, :, None]
        compare_img = np.concatenate([img_cv, masked_img], axis=1)
        ##--------------------------
        filename = os.path.basename(img_path)
        if save_root:
            cv2.imwrite(os.path.join(save_root, '{}.png'.format(filename.split('.')[0])), save_mask)
        if compare_root:
            cv2.imwrite(os.path.join(compare_root, '{}.jpg'.format(filename.split('.')[0])), compare_img)


def copy_and_rename(image_root, save_root):
    """
    resave val2019 image and order by number
    Args:
        image_root:
        save_root:

    Returns: None

    """
    if save_root:
        os.makedirs(save_root, exist_ok=True)
    filenames = [f for f in os.listdir(image_root) if os.path.isfile(os.path.join(image_root, f))]
    filenames.sort()
    index = 1
    for filename in filenames:
        shutil.copyfile(os.path.join(image_root, filename), os.path.join(save_root, '{}.jpg'.format(index)))
        index += 1


def bbox_2_area(json_path):
    with open(json_path) as fid:
        data = json.load(fid)
    # data_coco = COCO(annotation_file=json_path)
    annotations = data['annotations']
    new_anns = []
    for ann in tqdm(annotations):
        x, y, w, h = ann['bbox']
        area = int(w) * int(h)
        ann['area'] = area
        new_anns.append(ann)
    data['annotations'] = new_anns
    with open(json_path, "w") as outfile:
        json.dump(data, outfile, )


def coco_remove_catIds(json_path, save_path='remove_cat.json', remove_catIds=None):
    """
    移除特定的類別，且包含該類別的影像也移除
    :param json_path:
    :param save_path:
    :param remove_catIds:
    :return:
    """
    remove_catIds = [int(i.split('_', 1)[0]) for i in remove_catIds]
    if remove_catIds is None:
        return
    with open(json_path, 'r') as fid:
        data = json.load(fid)
    assert data is not None
    annotations = data['annotations']
    images = data['images']
    new_annotations = list()
    new_images = list()
    remove_images = list()
    ## remove annotation
    for i in range(len(annotations)):
        if annotations[i]['category_id'] in remove_catIds:
            remove_images.append(annotations[i]['image_id'])
        else:
            new_annotations.append(annotations[i])
    remove_images = list(set(remove_images))
    for i in range(len(images)):
        if images[i]['id'] not in remove_images:
            new_images.append(images[i])
    for ann in new_annotations:
        if ann['image_id'] in remove_images:
            new_annotations.remove(ann)
    new_annotations = [i for i in new_annotations if i['image_id'] not in remove_images]
    data['annotations'] = new_annotations
    data['images'] = new_images
    with open(save_path, "w") as outfile:
        json.dump(data, outfile, )


def get_avg_brightness(folder):
    brightness = []
    files = os.listdir(folder)
    for file in tqdm(files):
        img = cv2.imread(os.path.join(folder, file))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        brightness.append(np.average(v))
    brightness = np.array(brightness)
    print(np.average(brightness))


def coco_strink_data(json_path, save_path, ratio):
    """

    :param json_path:
    :param save_path:
    :param ratio: remove portion
    :return:
    """
    # TODO
    with open(json_path, 'r') as fid:
        data = json.load(fid)
    assert data is not None
    annotations = data['annotations']
    images = data['images']
    new_annotations = list()
    new_images = list()
    remove_images = random.choices(images, k=int(len(images) * ratio))
    for i in range(len(images)):
        if images[i]['id'] not in remove_images:
            new_images.append(images[i])
    for ann in annotations:
        if ann['image_id'] not in remove_images:
            new_annotations.append(ann)
    data['annotations'] = new_annotations
    data['images'] = new_images
    with open(save_path, "w") as outfile:
        json.dump(data, outfile, )


if __name__ == "__main__":
    print("...main...")
    # find_items()
    # get_sizeof_product()
    # extract_val_obj()
    # check_each_class_instance_count()
    ## ---------------------------------------
    # create_synthesys_shadow2('D:\\datasets\\rpc_list\\synthesize_10000',
    #                          'D:\\datasets\\rpc_list\\synthesize_10000_masks',
    #                          'D:\\datasets\\rpc_list\\synthesize_10000_shadows')
    ## ---------------------------------------
    # files = glob.glob('./bg*.jpg')
    # resize_background_images(files)
    ## ---------------------------------------
    # check_ann_duplicate_id('/media/ian/WD/datasets/rpc_list/sod_synthesize_15000_0.json')
    ## ---------------------------------------
    ### resize coco dataset (include json file, masks and images)
    config = Config()
    dataroot = config.get_dataset_root()

    name = 'synthesize_24000_train'
    image_folder = os.path.join(dataroot, 'rpc_list/{}_shadow'.format(name))
    mask_folder = None
    json_file = os.path.join(dataroot, 'rpc_list/{}.json'.format(name))
    target_image_folder = os.path.join(dataroot, 'rpc_list/{}_shadow_512'.format(name))
    target_mask_folder = None
    target_json_file = os.path.join(dataroot, 'rpc_list/{}_512.json'.format(name))
    rescale_coco_data(image_folder=image_folder,
                      mask_folder=mask_folder,
                      json_file=json_file,
                      target_size=512,
                      target_image_folder=target_image_folder,
                      target_mask_folder=target_mask_folder,
                      target_json_file=target_json_file)

    # image_folder = os.path.join(dataroot, 'retail_product_checkout/test2019')
    # mask_folder = None
    # json_file = os.path.join(dataroot, 'retail_product_checkout/instances_test2019.json')
    # target_image_folder = os.path.join(dataroot, 'retail_product_checkout/test2019_512')
    # target_mask_folder = None
    # target_json_file = os.path.join(dataroot, 'retail_product_checkout/instances_test2019_512.json')
    # rescale_coco_data(image_folder=image_folder,
    #                   mask_folder=mask_folder,
    #                   json_file=json_file,
    #                   target_size=512,
    #                   target_image_folder=target_image_folder,
    #                   target_mask_folder=target_mask_folder,
    #                   target_json_file=target_json_file)
    ## --------------------------------------
    # json_path = f'/media/ian/WD/datasets/rpc_list/synthesize_15000_best.json'
    # img_path = f'/media/ian/WD/datasets/rpc_list/synthesize_15000_best'
    # save_path = f'/media/ian/WD/datasets/rpc_list/synthesize_15000_best_labeled'
    # np.random.seed(42)
    # _COLORS = np.random.rand(200, 3)
    # visualize_bbox(json_path=json_path, img_path=img_path, save_path=save_path, COLORS=_COLORS, CLASSES=RPC_CLASSES)
    ## ----------------------------------------
    # mask = f'/media/ian/WD/datasets/rpc_list/synthesize_5_mix_mask_small/synthesized_image_0.png'
    # mask_np = cv2.imread(mask, cv2.IMREAD_COLOR)
    # print(np.unique(mask_np))
    ## ----------------------------------------
    # check_coco_seg_format('/media/ian/WD/datasets/rpc_list/synthesize_15000_test_small_seg.json')
    ## ----------------------------------------
    # check_ratio('./ratio_annotations_all.json')
    ## ----------------------------------------
    # get_sizeof_product()
    ## ----------------------------------------
    # add_area_2_bbox('/media/ian/WD/datasets/retail_product_checkout/annotations/instances_smallval2019.json')
    ## ----------------------------------------
    # extract_val_ann('/media/ian/WD/datasets/retail_product_checkout/instances_test2019.json',
    #                 '/media/ian/WD/datasets/retail_product_checkout/test2019',
    #                 '/media/ian/WD/datasets/retail_product_checkout/test2019_anns')
    ## ----------------------------------------
    # extract_ann_mask(image_root='/media/ian/WD/datasets/retail_product_checkout/test2019_anns',
    #                  save_root='/media/ian/WD/datasets/retail_product_checkout/test2019_anns_mask',
    #                  compare_root='/media/ian/WD/datasets/retail_product_checkout/test2019_anns_compare')
    ## ----------------------------------------
    # create_syn_val(json_path='/media/ian/WD/datasets/retail_product_checkout/instances_test2019.json',
    #                image_root='/media/ian/WD/datasets/retail_product_checkout/test2019',
    #                ann_root='/media/ian/WD/datasets/retail_product_checkout/test2019_anns',
    #                mask_root='/media/ian/WD/datasets/retail_product_checkout/test2019_anns_mask',
    #                save_root='/media/ian/WD/datasets/retail_product_checkout/test2019_syn')
    ## ----------------------------------------
    # copy_and_rename(image_root='/media/ian/WD/datasets/retail_product_checkout/test2019',
    #                 save_root='/media/ian/WD/datasets/rpc_transfer3/trainB')
    # copy_and_rename(image_root='/media/ian/WD/datasets/retail_product_checkout/test2019_syn',
    #                 save_root='/media/ian/WD/datasets/rpc_transfer3/trainA')
    ## ----------------------------------------
    # bbox_2_area(json_path='D:\\datasets\\retail_product_checkout\\instances_val2019_small.json')
    ## ----------------------------------------
    # coco_remove_catIds(json_path='D:/datasets/retail_product_checkout/instances_val2019.json',
    #                    save_path='D:/datasets/retail_product_checkout/instances_val2019_removed.json',
    #                    remove_catIds=['1_puffed_food', '13_dried_fruit', '22_dried_food', '31_instant_drink',
    #                                   '42_instant_noodles', '54_dessert', '71_drink', '79_alcohol', '81_drink',
    #                                   '88_alcohol', '97_milk', '108_canned_food', '122_chocolate', '134_gum',
    #                                   '142_candy',
    #                                   '152_seasoner', '164_personal_hygiene', '174_tissue', '194_stationery'])
    ## ------------------------------------------------------------
    # get_avg_brightness("D:/datasets/rpc_list/synthesize_30000_noshad")
    # get_avg_brightness("D:/datasets/retail_product_checkout/test2019")
