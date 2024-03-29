import argparse
import os
import time
import json
import cv2
from tqdm import tqdm
import numpy as np
import sys
from pathlib import Path
from config import Config

"""
為測試資料集的影像加上標籤(bbox)(在畫面中)
"""
COCO_CLASSES = ('1_puffed_food', '2_puffed_food', '3_puffed_food', '4_puffed_food', '5_puffed_food',
                '6_puffed_food', '7_puffed_food',
                '8_puffed_food', '9_puffed_food', '10_puffed_food', '11_puffed_food', '12_puffed_food',
                '13_dried_fruit',
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
                '91_alcohol', '92_alcohol', '93_alcohol', '94_alcohol', '95_alcohol', '96_alcohol', '97_milk',
                '98_milk',
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
                '189_tissue', '190_tissue', '191_tissue', '192_tissue', '193_tissue', '194_stationery',
                '195_stationery',
                '196_stationery', '197_stationery',
                '198_stationery', '199_stationery', '200_stationery')

np.random.seed(42)
_COLORS = np.random.rand(200, 3)

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def make_parser():
    parser = argparse.ArgumentParser("visualize coco format dataset!")
    parser.add_argument("--use_default", default=True, help="use default value? true or false")
    parser.add_argument("--img_path", default=None, help="path to images or video")
    parser.add_argument("--json_path", default=None, help="path to json file")
    parser.add_argument("--save_path", default=None, help="path to save file")
    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


def vis(img, box, cls_id, class_names=None):
    cls_id = cls_id - 1

    x0 = int(box[0])
    y0 = int(box[1])
    x1 = x0 + int(box[2])
    y1 = y0 + int(box[3])

    color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
    text = '{}'.format(class_names[cls_id], )
    txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX

    txt_size = cv2.getTextSize(text, font, 0.5, 1)[0]
    cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

    txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
    cv2.rectangle(
        img,
        (x0, y0 + 1),
        (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])),
        txt_bk_color,
        -1
    )
    cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.5, txt_color, thickness=1)

    return img


def main(args):
    print("__main__")
    Path.mkdir(Path(args.save_path).resolve(), parents=True, exist_ok=True)

    with open(args.json_path, 'rb') as json_file:
        json_data = json.load(json_file)
    imgs = json_data['images']
    anns = json_data['annotations']
    anns = sorted(anns, key=lambda i: (i['image_id'], i['id']))
    image_id_2_filename = {img['id']: img['file_name'] for img in imgs}
    current_image_id = None
    img_cv = None
    # idx = 0
    for ann in anns:
        if current_image_id != ann['image_id']:
            if current_image_id is not None:
                cv2.imwrite(os.path.join(args.save_path, image_id_2_filename[current_image_id]), img_cv)
            current_image_id = ann['image_id']
            img_cv = cv2.imread(os.path.join(args.img_path, image_id_2_filename[current_image_id]))
            vis(img_cv, ann['bbox'], ann['category_id'], COCO_CLASSES)
        else:
            vis(img_cv, ann['bbox'], ann['category_id'], COCO_CLASSES)
        # idx += 1
        # if idx == 10:
        #     break


if __name__ == "__main__":
    cfg = Config()
    DATASET_DIR = cfg.get_dataset_root()
    args = make_parser().parse_args()
    if args.use_default:
        args.img_path = os.path.join(DATASET_DIR, 'retail_product_checkout', 'smallval2019')
        args.json_path = os.path.join(DATASET_DIR, 'retail_product_checkout', 'instances_smallval2019.json')
        args.save_path = os.path.join(DATASET_DIR, 'retail_product_checkout', 'smallval2019_labeled')
    main(args)
