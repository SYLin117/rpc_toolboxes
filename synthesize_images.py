import glob
import json
import os
import random
import scipy
import scipy.spatial as T
import time
from argparse import ArgumentParser
from collections import defaultdict
import cv2
import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage
from tqdm import tqdm
from config import Config
import pathlib
import sys
import matplotlib.pyplot as plt

NUM_CATEGORIES = 200
GENERATED_NUM = 100

CATEGORIES = ['__background__', '1_puffed_food', '2_puffed_food', '3_puffed_food', '4_puffed_food', '5_puffed_food',
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
              '198_stationery', '199_stationery', '200_stationery']

np.random.seed(42)
CAT_COLORS = (np.random.rand(201, 3) * 255).astype(np.uint8)
CAT_COLORS[0, :] = [0, 0, 0]


def buy_strategic(counter):
    categories = [i + 1 for i in range(NUM_CATEGORIES)]
    selected_categories = np.random.choice(categories, size=random.randint(3, 10), replace=False)
    num_categories = len(selected_categories)

    if 3 <= num_categories < 5:  # Easy mode: 3∼5
        num_instances = random.randint(num_categories, 10)
        counter['easy_mode'] += 1
    elif 5 <= num_categories < 8:  # Medium mode: 5∼8
        num_instances = random.randint(10, 15)
        counter['medium_mode'] += 1
    else:  # Hard mode: 8∼10
        num_instances = random.randint(15, 20)
        counter['hard_mode'] += 1

    num_per_category = {}
    generated = 0
    for i, category in enumerate(selected_categories):
        i += 1
        if i == num_categories:
            count = num_instances - generated
        else:
            count = random.randint(1, num_instances - (num_categories - i) - generated)
        generated += count
        num_per_category[int(category)] = count

    return num_per_category


def check_iou(annotations, box, threshold=0.5):
    """
    Args:
        annotations:
        box: (x, y, w, h)
        threshold:
    Returns: bool
    """

    cx1, cy1, cw, ch = box
    cx2, cy2 = cx1 + cw, cy1 + ch
    carea = cw * ch  # new object
    for ann in annotations:
        x1, y1, w, h = ann['bbox']
        x2, y2 = x1 + w, y1 + h
        area = w * h  # object in ann
        inter_x1 = max(x1, cx1)
        inter_y1 = max(y1, cy1)
        inter_x2 = min(x2, cx2)
        inter_y2 = min(y2, cy2)

        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        iou = inter_area / (carea + area - inter_area + 1e-8)  # avoid division by zero

        iou1 = inter_area / (carea + 1e-8)  # 重疊區域佔舊object的比例
        iou2 = inter_area / (area + 1e-8)  # 重疊區域佔新object的比例

        if iou > threshold or iou1 > threshold or iou2 > threshold:
            return False
    return True


def sample_select_object_index(category, paths, ratio_annotations, threshold=0.1):
    """
    randomly choose one file that match threshold
    Args:
        category:
        paths:
        ratio_annotations:
        threshold:

    Returns:

    """
    high_threshold_paths = [path for path in paths if ratio_annotations[os.path.basename(path)] > threshold]
    index = random.randint(0, len(high_threshold_paths) - 1)
    path = high_threshold_paths[index]
    return path


def generated_position(width, height, w, h, pad=0):
    x = random.randint(pad, width - w - pad)
    y = random.randint(pad, height - h - pad)
    return x, y


def get_object_bbox(annotation, max_width, max_height):
    bbox = annotation['bbox']
    x, y, w, h = [int(x) for x in bbox]

    # box_pad = max(160, int(max(w, h) * 0.3))
    box_pad = 5
    crop_x1 = max(0, x - box_pad)
    crop_y1 = max(0, y - box_pad)
    crop_x2 = min(x + w + box_pad, max_width)
    crop_y2 = min(y + h + box_pad, max_height)
    x = crop_x1
    y = crop_y1
    w = crop_x2 - crop_x1
    h = crop_y2 - crop_y1
    return x, y, w, h


def gaussian_filter_density(gt):
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density
    pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))  # (x,y)
    leaf_size = 2048
    # build kd tree
    tree = T.KDTree(pts.copy(), leafsize=leaf_size)
    # query kd tree
    distances, locations = tree.query(pts, k=4)

    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1], pt[0]] = 1.
        if gt_count > 1:
            sigma = (distances[i][1] + distances[i][2] + distances[i][3]) * 0.085
            sigma = min(sigma, 999)  # avoid inf
        else:
            raise NotImplementedError('should not be here!!')
        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
    return density


def synthesize(strategics, save_json_file='', output_dir='', save_mask=False, train_json=None, train_imgs_dir=None,
               train_imgs_mask_dir=None, file_filter=None):
    with open('ratio_annotations.json') as fid:
        ratio_annotations = json.load(fid)

    with open(train_json) as fid:
        data = json.load(fid)
    images = {}
    for x in data['images']:
        images[x['id']] = x

    annotations = {}
    for x in data['annotations']:
        annotations[images[x['image_id']]['file_name']] = x

    # object_paths = glob.glob(
    #     os.path.join('/media/ian/WD/PythonProject/DP-Net/acm-mm-2019-ACO-master/toolboxes/extracted_masks/crop_images/',
    #                  '*.jpg'))
    object_paths = list()
    if not file_filter:
        object_paths = glob.glob(os.path.join(train_imgs_dir, '*.jpg'))
    else:
        for filter in file_filter:
            object_paths += glob.glob(os.path.join(train_imgs_dir, filter))

    object_category_paths = defaultdict(list)
    for path in object_paths:
        name = os.path.basename(path)
        category = annotations[name]['category_id']
        object_category_paths[category].append(path)
    object_category_paths = dict(object_category_paths)  # store each categories all single images

    bg_img_cv = cv2.imread('bg.jpg')
    bg_height, bg_width = bg_img_cv.shape[:2]
    mask_img_cv = np.zeros((bg_height, bg_width), dtype=np.uint8)

    json_ann = []
    json_img = []
    ann_idx = 0
    for image_id, num_per_category in tqdm(strategics):
        img_id_num = image_id.split('_')[2]
        bg_img = Image.fromarray(bg_img_cv)
        mask_img = Image.fromarray(mask_img_cv).convert('RGB')
        obj_in_this_pic = list()
        for category, count in num_per_category.items():
            category = int(category)
            for _ in range(count):
                paths = object_category_paths[category]

                object_path = sample_select_object_index(category, paths, ratio_annotations, threshold=0.1)

                name = os.path.basename(object_path)
                mask_path = os.path.join(train_imgs_mask_dir, '{}.png'.format(name.split('.')[0]))

                obj = Image.open(object_path)
                mask = Image.open(mask_path).convert('L')
                original_width = obj.width
                original_height = obj.height
                # dense object bbox
                # ---------------------------
                # Crop according to json annotation
                # ---------------------------
                x, y, w, h = get_object_bbox(annotations[name], original_width, original_height)
                obj = obj.crop((x, y, x + w, y + h))
                mask = mask.crop((x, y, x + w, y + h))

                # ---------------------------
                # Random scale
                # ---------------------------
                scale = random.uniform(0.5, 0.7)
                w, h = int(w * scale), int(h * scale)
                obj = obj.resize((w, h), resample=Image.BILINEAR)
                mask = mask.resize((w, h), resample=Image.BILINEAR)

                # ---------------------------
                # Random rotate
                # ---------------------------
                angle = random.random() * 360
                obj = obj.rotate(angle, resample=Image.BILINEAR, expand=1)
                mask = mask.rotate(angle, resample=Image.BILINEAR, expand=1)

                # ---------------------------
                # Crop according to mask
                # ---------------------------
                where = np.where(np.array(mask))  # value == 255 location
                # where = np.vstack((where[0], where[1]))  ## ian added
                assert len(where[0]) != 0
                assert len(where[1]) != 0
                assert len(where[0]) == len(where[1])
                area = len(where[0])
                y1, x1 = np.amin(where, axis=1)
                y2, x2 = np.amax(where, axis=1)

                obj = obj.crop((x1, y1, x2, y2))
                mask = mask.crop((x1, y1, x2, y2))
                w, h = obj.width, obj.height

                pad = 2
                pos_x, pos_y = generated_position(bg_width, bg_height, w, h, pad)
                start = time.time()
                threshold = 0.25
                while not check_iou(obj_in_this_pic, box=(pos_x, pos_y, w, h), threshold=threshold):
                    if (time.time() - start) > 3:  # cannot find a valid position in 3 seconds
                        start = time.time()
                        threshold += 0.05
                        continue
                    pos_x, pos_y = generated_position(bg_width, bg_height, w, h, pad)

                bg_img.paste(obj, box=(pos_x, pos_y), mask=mask)
                if save_mask:
                    color_mask = Image.new('RGB', mask.size, tuple(CAT_COLORS[category])) # 挑到mask畫面上的物件mask
                    color_mask_cv2 = np.asarray(color_mask)
                    mask_cv2 = np.asarray(mask)
                    kernel = np.ones((3, 3), np.uint8)
                    erosion = cv2.erode(mask_cv2, kernel, iterations=3)
                    # _, mask_cv2 = cv2.threshold(mask_cv2, 127, 255, cv2.THRESH_BINARY_INV)
                    Contours, _ = cv2.findContours(erosion, cv2.RETR_EXTERNAL,
                                                   cv2.CHAIN_APPROX_TC89_KCOS)
                    # mask_cv2_rgb = cv2.cvtColor(mask_cv2, cv2.COLOR_GRAY2RGB, )
                    for cnt in Contours:
                        hull = cv2.convexHull(cnt)
                        cv2.drawContours(color_mask_cv2, [hull], -1, color=(0, 0, 0), thickness=4, )

                    plt.imshow(color_mask_cv2)
                    plt.show()
                    plt.close()
                    color_mask = Image.fromarray(color_mask_cv2)
                    mask_img.paste(color_mask, box=(pos_x, pos_y), mask=mask)
                # plt.imshow(mask)
                # plt.show()

                # ---------------------------
                # Find center of mass
                # ---------------------------
                mask_array = np.array(mask)
                center_of_mass = ndimage.measurements.center_of_mass(mask_array)  # y, x
                center_of_mass = [int(round(x)) for x in center_of_mass]
                center_of_mass = center_of_mass[1] + pos_x, center_of_mass[0] + pos_y  # map to whole image

                json_ann.append({
                    'bbox': (pos_x, pos_y, w, h),
                    'category_id': category,
                    'center_of_mass': center_of_mass,
                    'area': area,
                    'image_id': int(img_id_num),
                    'iscrowd': 0,
                    'id': ann_idx
                })
                obj_in_this_pic.append({
                    'bbox': (pos_x, pos_y, w, h),
                    'category_id': category,
                    'center_of_mass': center_of_mass,
                    'area': area,
                    'image_id': int(img_id_num),
                    'iscrowd': 0,
                    'id': ann_idx
                })
                ann_idx += 1

        assert bg_height == 1815 and bg_width == 1815
        scale = 200.0 / 1815
        gt = np.zeros((round(bg_height * scale), round(bg_width * scale)))
        for item in obj_in_this_pic:
            center_of_mass = item['center_of_mass']
            gt[round(center_of_mass[1] * scale), round(center_of_mass[0] * scale)] = 1

        assert gt.shape[0] == 200 and gt.shape[1] == 200

        # density = gaussian_filter_density(gt)  # gussian to gt
        image_name = '{}.jpg'.format(image_id)

        bg_img.save(os.path.join(output_dir, image_name))
        # np.save(os.path.join(output_dir, 'density_maps', image_id), density) # save density

        # plt.subplot(121)
        # plt.imshow(density, cmap='gray')
        #
        # plt.subplot(122)
        # plt.imshow(bg_img)
        #
        # print(len(synthesize_annotations))
        # print(density.sum())
        # plt.show()
        # quit()

        if save_mask:
            mask_img.save(os.path.join(output_dir, 'masks', image_name))
        # json_ann.append({
        #     'image_id': image_name,
        #     'objects': synthesize_annotations
        # })

        json_img.append({
            'file_name': image_name,
            'id': int(img_id_num),
            'width': 1815,
            'height': 1815,
        })
    json_cat = list()
    for idx, val in enumerate(CATEGORIES):
        if idx == 0:
            continue
        name_split = val.split('_', 1)
        id = name_split[0]
        super_cat = name_split[1]
        json_cat.append({'supercategory': super_cat, 'id': int(id), 'name': val})
    new_json = dict()
    new_json['images'] = json_img
    new_json['annotations'] = json_ann
    new_json['categories'] = json_cat
    if save_json_file:
        with open(save_json_file, 'w') as fid:
            json.dump(new_json, fid)


if __name__ == '__main__':
    parser = ArgumentParser(description="Synthesize fake images")
    parser.add_argument('--count', type=int, default=1)  ## original=32
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    ###########################################################################################
    ###########################################################################################
    counter = {
        'easy_mode': 0,
        'medium_mode': 0,
        'hard_mode': 0
    }
    strategics = []
    for image_id in tqdm(range(GENERATED_NUM)):
        num_per_category = buy_strategic(counter)
        strategics.append(('synthesized_image_{}'.format(image_id), num_per_category))
    strategics_name = 'strategics_train.json'
    if os.path.exists(strategics_name):
        os.remove(strategics_name)
    with open(strategics_name, 'w') as f:
        json.dump(strategics, f)
    print(counter)  # {'easy_mode': 25078, 'medium_mode': 37287, 'hard_mode': 37635}
    # quit()
    ###########################################################################################
    ###########################################################################################
    with open(strategics_name) as f:
        strategics = json.load(f)
    strategics = sorted(strategics, key=lambda s: s[0])
    version = str(GENERATED_NUM)

    output_dir = os.path.join('synthesize_{}'.format(version))
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if not os.path.exists(os.path.join(output_dir, 'density_maps')):
        os.mkdir(os.path.join(output_dir, 'density_maps'))

    if not os.path.exists(os.path.join(output_dir, 'masks')):
        os.mkdir(os.path.join(output_dir, 'masks'))

    threads = []
    num_threads = args.count
    sub_strategics = strategics[args.local_rank::num_threads]
    config = Config()
    DATASET_ROOT = config.get_dataset_root()
    CURRENT_ROOT = str(pathlib.Path().resolve())
    save_file = os.path.join(sys.path[0],
                             'sod_synthesize_{}_{}.json'.format(version, args.local_rank))  # synthesis images json檔案
    rpc_train_json = os.path.join(DATASET_ROOT, 'retail_product_checkout',
                                  'instances_train2019.json')  # rpc的原始train.json
    rpc_train = os.path.join(DATASET_ROOT, 'retail_product_checkout', 'train2019')  # rpc 的train影像資料夾
    rpc_train_mask = os.path.join(CURRENT_ROOT, 'extracted_masks_tracer5_morph10', 'masks')  # 擷取的mask影像資料夾
    synthesize(sub_strategics, save_file, output_dir,
               train_json=rpc_train_json,
               train_imgs_dir=rpc_train,
               train_imgs_mask_dir=rpc_train_mask,
               file_filter=config.img_filters,
               save_mask=True)
