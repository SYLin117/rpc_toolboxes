# ==========================
# 使用multithread製作偽造影像
# specify object order in image
# ==========================
import glob
import json
import os
import random
from random import randrange
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
import multiprocessing
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from bounded_pool_executor import BoundedThreadPoolExecutor
import gc
import traceback
from scipy.stats import truncnorm
import math


def get_truncated_normal(mean=0.25, sd=0.05, low=0.0, upp=1.0):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


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
CAT_COLORS = (1 - (np.random.rand(201, 3)) * 255).astype(np.uint8)
CAT_COLORS[0, :] = [0, 0, 0]
SHADOW_COLOR = [0x403e3b, 0x454340]


def buy_strategic(counter):
    """

    Args:
        counter:

    Returns:num_per_category, difficulty

    """
    global NUM_CATEGORIES
    categories = [i + 1 for i in range(NUM_CATEGORIES)]
    difficulty = random.randint(1, 3)
    if difficulty == 1:
        num_categories = random.randint(3, 5)
        num_instances = random.randint(num_categories, 5)
        counter['easy_mode'] += 1
    elif difficulty == 2:
        num_categories = random.randint(5, 8)
        num_instances = random.randint(num_categories, 8)
        counter['medium_mode'] += 1
    elif difficulty == 3:
        num_categories = random.randint(8, 10)
        num_instances = random.randint(num_categories, 10)
        counter['hard_mode'] += 1
    num_per_category = {}
    generated = 0
    selected_categories = np.random.choice(categories, size=num_categories, replace=False)
    # ================= original rule ==========================
    # for i, category in enumerate(selected_categories):
    #     i += 1
    #     if i == num_categories:
    #         count = num_instances - generated
    #     else:
    #         count = random.randint(1, num_instances - (num_categories - i) - generated)
    #
    #     generated += count
    #     num_per_category[int(category)] = count
    for category in selected_categories:
        count = random.randint(1, 3)
        num_per_category[int(category)] = count
    return num_per_category, difficulty


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


def sample_select_object_index(category, paths, ratio_annotations, threshold=0.5):
    """
    randomly choose one file that match threshold
    Args:
        paths: 該物件的影像(list)
        ratio_annotations: 儲存所有檔案的ratio資料
        threshold: 滿足的threshold

    Returns:

    """
    high_threshold_paths = list()
    # if category in [160, 161, 162, 163]:  # 瓶狀的調味品
    #     ratio_list = [ratio_annotations[os.path.basename(path)] for path in paths]
    #     high_threshold_paths.append(paths[ratio_list.index(max(ratio_list))])
    # else:
    #     high_threshold_paths = [path for path in paths if ratio_annotations[os.path.basename(path)] > threshold]
    high_threshold_paths = [path for path in paths if ratio_annotations[os.path.basename(path)] > threshold]
    while len(high_threshold_paths) == 0:  # 如果沒有>threshold的圖片 則取最大的
        threshold -= 0.1
        high_threshold_paths = [path for path in paths if ratio_annotations[os.path.basename(path)] > threshold]
        # ratio_list = [ratio_annotations[os.path.basename(path)] for path in paths]
        # high_threshold_paths.append(paths[ratio_list.index(max(ratio_list))])
    # high_threshold_paths = paths
    index = random.randint(0, len(high_threshold_paths) - 1)
    path = high_threshold_paths[index]
    return path


def generated_position(width, height, w, h, padx=0, pady=0):
    x = random.randint(padx, width - w - padx)
    y = random.randint(pady, height - h - pady)
    while x + w > width:
        x = random.randint(padx, width - w - padx)
    while y + h > height:
        y = random.randint(pady, height - h - pady)
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


def trans_paste(bg_img: np.ndarray, fg_img: np.ndarray, mask: np.ndarray, bbox: tuple, trans: bool):
    pos_x, pos_y, w, h = bbox
    try:
        if trans:  # fg_img: shadow
            shadow_prop = random.uniform(0.3, 0.7)
            bg_img[pos_y:pos_y + h, pos_x:pos_x + w, :] = \
                bg_img[pos_y:pos_y + h, pos_x:pos_x + w, :] * (np.ones_like(mask) - mask) + \
                fg_img * mask * shadow_prop + bg_img[pos_y:pos_y + h, pos_x:pos_x + w, :] * mask * (1 - shadow_prop)
        else:
            bg_img[pos_y:pos_y + h, pos_x:pos_x + w, :] = \
                bg_img[pos_y:pos_y + h, pos_x:pos_x + w, :] * (np.ones_like(mask) - mask) + fg_img * mask
    except ValueError as ve:
        print(str(ve))
    return bg_img


def get_random_pos_neg():
    """
    randomly return 1 or -1
    Returns: 1 or -1

    """
    return 1 if random.random() < 0.8 else -1


def create_image(image_id, num_per_category, change_background: bool, paste_shadow:bool, lock: Lock):
    try:
        global level_dict, ann_idx, json_ann, json_img, json_color  # ann_idx: for create new annotation id, json_ann: new annotation list, json_img: new image list
        instance_num = 0
        for category, count in num_per_category.items():
            instance_num += count
        INSTANCE_COLOR = (1 - (np.random.rand(instance_num, 3)) * 255).astype(np.uint8)
        instance_id = 0
        color_cat_dict = {}
        color_annId_dict = {}
        # ----------------- get background image --------------------
        background_id = random.randint(1, 3)
        if not change_background:
            background_id = 1
        bg_img_cv = cv2.imread('bg{}.jpg'.format(background_id), cv2.IMREAD_COLOR)
        bg_img_cv = cv2.cvtColor(bg_img_cv, cv2.COLOR_BGR2RGB)
        bg_height, bg_width = bg_img_cv.shape[:2]
        mask_img_cv = np.zeros((bg_height, bg_width, 3), dtype=np.uint8)
        # ----------------- get background image --------------------
        img_id_num = image_id.split('_')[2]
        bg_img = Image.fromarray(bg_img_cv).convert("RGB")
        mask_img = Image.fromarray(mask_img_cv)
        mask_img_np = mask_img_cv.copy()  # mask background image
        obj_in_this_pic = list()
        for category, count in num_per_category.items():
            category = int(category)
            for _ in range(count):
                paths = object_category_paths[category]

                object_path = sample_select_object_index(category, paths, ratio_annotations, threshold=0.45)

                name = os.path.basename(object_path)
                mask_path = os.path.join(train_imgs_mask_dir, '{}.png'.format(name.split('.')[0]))

                obj = Image.open(object_path)
                mask = Image.open(mask_path).convert('1')
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
                # scale = 1
                # ===============================================================================
                # scale_mean = train_val_ratio[str(category)]
                # scale = get_truncated_normal(mean=scale_mean, sd=0.05, low=.0, upp=2.0).rvs()
                # while scale <= 0:
                #     scale = get_truncated_normal(mean=scale_mean-.1, sd=0.05, low=.0, upp=2.0).rvs()
                # ===============================================================================
                scale_mean = train_val_ratio[str(category)]
                scale_mean = math.sqrt(scale_mean)
                if category in ([i for i in range(71, 122)] + [i for i in range(160, 164)]):  # drink
                    scale_mean -= 0.15
                if category in ([i for i in range(136, 142)] + [i for i in range(145, 147)]):  # small gum
                    scale_mean += 0.15
                std = 0.01
                low = scale_mean - 3 * std
                up = scale_mean + 3 * std
                scale = get_truncated_normal(mean=scale_mean, sd=std, low=low, upp=up).rvs()
                while scale <= 0:
                    scale = get_truncated_normal(mean=scale_mean, sd=std, low=low, upp=up).rvs()
                w, h = int(w * scale), int(h * scale)
                obj = obj.resize((w, h), resample=Image.BILINEAR)
                mask = mask.resize((w, h), resample=Image.BILINEAR)

                # ---------------------------
                # Random rotate
                # ---------------------------
                angle = random.random() * 360
                obj = obj.rotate(angle, resample=Image.BILINEAR, expand=True)
                mask = mask.rotate(angle, resample=Image.BILINEAR, expand=True)

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
                mask_l = mask.convert('L')
                w, h = obj.width, obj.height
                offset = []
                if paste_shadow:
                    offset.append(np.random.randint(5, 15) * get_random_pos_neg())  # right offset
                    offset.append(np.random.randint(10, 40) * get_random_pos_neg())  # down offset
                else:
                    offset.append(0)
                    offset.append(0)

                pos_x, pos_y = generated_position(bg_width, bg_height, w, h, padx=abs(offset[0]), pady=abs(offset[1]))
                start = time.time()
                threshold = 0.2
                while not check_iou(obj_in_this_pic, box=(pos_x, pos_y, w, h), threshold=threshold):
                    if (time.time() - start) > 1:  # cannot find a valid position in 3 seconds
                        start = time.time()
                        threshold += 0.05
                        continue
                    pos_x, pos_y = generated_position(bg_width, bg_height, w, h, padx=abs(offset[0]),
                                                      pady=abs(offset[1]))

                obj_cv = np.array(obj)
                mask_cv = np.array(mask) * 1  # single channel mask
                mask_cv = np.stack((mask_cv, mask_cv, mask_cv), axis=2)  # RGB mask
                shadow_indx = random_index = randrange(len(SHADOW_COLOR))
                shadow = Image.new('RGB', (w, h), SHADOW_COLOR[shadow_indx])
                shodow_cv = np.array(shadow)

                if paste_shadow:
                    trans_paste(bg_img_cv, shodow_cv, mask_cv, bbox=(pos_x + offset[0], pos_y + offset[1], w, h),
                                trans=True)
                trans_paste(bg_img_cv, obj_cv, mask_cv, bbox=(pos_x, pos_y, w, h), trans=False)
                # bg_img.paste(obj, box=(pos_x, pos_y), mask=mask)

                # plt.imshow(mask)
                # plt.show()

                # ---------------------------
                # Find center of mass
                # ---------------------------
                mask_array = np.array(mask)
                center_of_mass = ndimage.measurements.center_of_mass(mask_array)  # y, x
                center_of_mass = [int(round(x)) for x in center_of_mass]
                center_of_mass = center_of_mass[1] + pos_x, center_of_mass[0] + pos_y  # map to whole image
                with lock:
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
                    if save_mask:
                        color_mask = Image.new('RGB', mask.size,
                                               tuple(INSTANCE_COLOR[instance_id].tolist()))  # 挑到mask畫面上的物件mask
                        color_tuple_list = INSTANCE_COLOR[instance_id].tolist()
                        color_tuple_list.reverse()
                        color_cat_dict[str(tuple(color_tuple_list))] = category
                        color_annId_dict[str(tuple(color_tuple_list))] = ann_idx

                        # ----------------- # using erode to create border for different mask ----------------------
                        # color_mask_cv2 = np.asarray(color_mask)
                        # mask_cv2 = np.asarray(mask)
                        # kernel = np.ones((3, 3), np.uint8)
                        # erosion = cv2.erode(mask_cv2, kernel, iterations=3)
                        # diff = cv2.absdiff(mask_cv2, erosion) / 255  # use as mask for black border paint on color mask
                        # diff = np.stack((diff, diff, diff), axis=2)  # make it to 3-channel
                        # blank_image = np.zeros((mask.size[1], mask.size[0], 3), np.uint8)
                        # # 為了讓相同的物件重疊的時候有黑邊可以區隔
                        # color_mask_cv2[:, :, :] = color_mask_cv2[:, :, :] * (1 - diff) + blank_image * diff
                        # color_mask = Image.fromarray(color_mask_cv2)
                        ## ---------------------------------------
                        # ## ------------ another way to make mask border
                        # ## 找出mask的邊界
                        # Contours, _ = cv2.findContours(erosion, cv2.RETR_EXTERNAL,
                        #                                cv2.CHAIN_APPROX_TC89_KCOS)
                        # for cnt in Contours:
                        #     hull = cv2.convexHull(cnt)
                        #     cv2.drawContours(color_mask_cv2, [hull], -1, color=(0, 0, 0), thickness=4, )
                        ## ---------------------------------------
                        # ------------------- using opencv not pil to paste the mask ---------------------
                        color_mask_np = np.array(color_mask)
                        mask_np = np.array(mask) * 1  # single channel mask
                        mask_np = np.stack((mask_np, mask_np, mask_np), axis=2)  # RGB mask
                        mask_img_np[pos_y:pos_y + h, pos_x:pos_x + w, :] = \
                            mask_img_np[pos_y:pos_y + h, pos_x:pos_x + w, :] * (np.ones_like(mask_np) - mask_np) + \
                            color_mask_np * mask_np
                        # mask_img = Image.fromarray(mask_img_np)
                        # plt.imshow(mask_img)
                        # plt.show()
                        # ----------------------------------------
                        # mask_img.paste(color_mask, box=(pos_x, pos_y), mask=mask)
                        # ----------------------------------------
                        instance_id += 1
                    ann_idx += 1
        # -------------------------------
        ## save image (mask) and json file
        # -------------------------------
        json_color[image_id] = {
            'color_dict': color_cat_dict,
            'color_annId_dict': color_annId_dict
        }
        image_name = '{}.jpg'.format(image_id)
        mask_name = '{}.png'.format(image_id)  # mask use png format, jpg would change pixel value after saving
        bg_img = Image.fromarray(bg_img_cv)
        bg_img.save(os.path.join(output_dir, image_name))
        if save_mask:
            # mask_img = Image.fromarray(mask_img_np)
            # print('mask contain :{} values'.format(np.unique(mask_img_np)))
            cv2.imwrite(os.path.join(mask_dir, mask_name), mask_img_np)
            # mask_img.save(os.path.join(mask_dir, image_name))
            # plt.imshow(mask_img)
            # plt.show()
            # time.sleep(100)
        json_img.append({
            'file_name': image_name,
            'id': int(img_id_num),
            'width': 1815,
            'height': 1815,
            'level': level_dict[os.path.basename(image_name).split('.')[0]]
        })
    except:
        traceback.print_exc()


def get_object_paths():
    object_paths = list()
    with open('cat_2_angle.json') as fid:
        cat_2_angle = json.load(fid)
    with open('product_code.json') as fid:
        product_code = json.load(fid)
    class_2_product_code = {}
    for k, v in product_code.items():
        class_2_product_code["{}_{}".format(v['cat_id'], v['sku_class'])] = k
    rotate_angles = [str(x) for x in range(1, 40, 1)]
    for code, value in product_code.items():
        # ========================================= custom angles
        angles = cat_2_angle[value['sku_class']]
        if value['cat_id'] in [i for i in range(160, 164)]:  # 瓶裝的調味料(seasoner)(其他是包裝的)
            angles = [0]
        elif value['cat_id'] in [i for i in range(164, 169)]:  # cylinder-like personal hygine
            angles = [0]
        elif value['cat_id'] in [i for i in range(169, 174)]:  # box-like personal hygine
            angles = [0, 2]
        elif value['cat_id'] in [37, 39, 40, 41]:  # 包裝類的即溶飲料(instant-drink)(其他是罐裝的)
            angles = [2]
        elif value['cat_id'] in [134, 135, 144]:  # canned candy(not package)
            angles = [0]
        elif value['cat_id'] in [45, 46, 47, 48, 49]:  # cup noodle
            angles = [0, 2]
        elif value['cat_id'] in [198, 200]:  # cylinder-like
            angles = [0]  # horizontal
        elif value['cat_id'] in [i for i in range(189, 194)]:  # small packet tissue
            angles = [2]  # top view
        elif value['cat_id'] in [i for i in range(71, 122)]:  # cylinder-like object(drink, alcohol)
            angles = [0]
        elif value['cat_id'] in [31, 32, 38]:
            angles = [0]
        # elif value['cat_id'] in [i for i in range(174, 189)]:  # big tissue
        #     angles = [0, 2]
        # =========================================
        # angles = [0, 1, 2, 3]
        # =========================================
        for ca in angles:  # camera angle
            for ra in rotate_angles:  # rotate angle
                object_paths += glob.glob(os.path.join(train_imgs_dir, '{}_camera{}-{}.jpg'.format(code, ca, ra)))
                object_paths += glob.glob(
                    os.path.join(train_imgs_dir, '{}-back_camera{}-{}.jpg'.format(code, ca, ra)))  # 有背面的商品
    # ================= remove specific rotation angle of tissue(too much white) ========================
    remove_paths = []
    nos = list(set([i for i in range(174, 189)]).difference({178}))  # large tissue, (178 shoot in different angle)
    remove_rotate_angles = [i for i in range(2, 21)] + [i for i in range(23, 41)]
    for no in nos:
        code = class_2_product_code["{}_tissue".format(no)]
        angles = [0, 1, 2, 3]
        for ca in angles:  # camera angle
            for ra in remove_rotate_angles:
                remove_paths += glob.glob(
                    os.path.join(train_imgs_dir, '{}_camera{}-{}.jpg'.format(code, ca, ra)))
                remove_paths += glob.glob(
                    os.path.join(train_imgs_dir, '{}-back_camera{}-{}.jpg'.format(code, ca, ra)))
    # =====================================================================================================
    nos = [i for i in range(189, 194)]  # small tissue, (178 shoot in different angle)
    remove_rotate_angles = [i for i in range(2, 41)]
    for no in nos:
        code = class_2_product_code["{}".format(CATEGORIES[no])]
        angles = [0, 1, 2, 3]
        for ca in angles:  # camera angle
            for ra in remove_rotate_angles:
                remove_paths += glob.glob(
                    os.path.join(train_imgs_dir, '{}_camera{}-{}.jpg'.format(code, ca, ra)))
                remove_paths += glob.glob(
                    os.path.join(train_imgs_dir, '{}-back_camera{}-{}.jpg'.format(code, ca, ra)))
    # =====================================================================================================
    nos = [178]  # large tissue, (178 shoot in different angle)
    front_remove_rotate_angles = [i for i in range(1, 11)] + [i for i in range(12, 31)] + [i for i in range(32, 41)]
    back_remove_rotate_angles = [i for i in range(2, 20)] + [i for i in range(21, 41)]
    for no in nos:
        code = class_2_product_code["{}".format(CATEGORIES[no])]
        angles = [0, 1, 2, 3]
        for ca in angles:  # camera angle
            for ra in front_remove_rotate_angles:
                remove_paths += glob.glob(
                    os.path.join(train_imgs_dir, '{}_camera{}-{}.jpg'.format(code, ca, ra)))
            for ra in back_remove_rotate_angles:
                remove_paths += glob.glob(
                    os.path.join(train_imgs_dir, '{}-back_camera{}-{}.jpg'.format(code, ca, ra)))
    # =====================================================================================================
    # box like object
    # only remain straight angle of object
    nos = [i for i in range(169, 174)]
    remove_rotate_angles = [i for i in range(2, 22)] + [i for i in range(23, 41)]
    for no in nos:
        code = class_2_product_code["{}".format(CATEGORIES[no])]
        angles = [0, 1, 2, 3]
        for ca in angles:  # camera angle
            for ra in remove_rotate_angles:
                remove_paths += glob.glob(
                    os.path.join(train_imgs_dir, '{}_camera{}-{}.jpg'.format(code, ca, ra)))
                remove_paths += glob.glob(
                    os.path.join(train_imgs_dir, '{}-back_camera{}-{}.jpg'.format(code, ca, ra)))
    # =====================================================================================================
    nos = [84, 85, 86, 87, 97, 98, ] \
          + [i for i in range(101, 108)] \
          + [31, 32, 38]  ## box like  drink, milk, box like instant drink
    remove_rotate_angles = list(set([i for i in range(1, 41)]).difference({1, 2, 11, 12, 20, 21, 30, 31}))
    for no in nos:
        code = class_2_product_code["{}".format(CATEGORIES[no])]
        angles = [0, 1, 2, 3]
        for ca in angles:  # camera angle
            for ra in remove_rotate_angles:
                remove_paths += glob.glob(
                    os.path.join(train_imgs_dir, '{}_camera{}-{}.jpg'.format(code, ca, ra)))
                remove_paths += glob.glob(
                    os.path.join(train_imgs_dir, '{}-back_camera{}-{}.jpg'.format(code, ca, ra)))
    # =====================================================================================================
    object_paths = list(set(object_paths).difference(set(remove_paths)))
    return object_paths


if __name__ == '__main__':
    parser = ArgumentParser(description="Synthesize fake images")
    parser.add_argument('--gen_num', type=int, default=30,
                        help='how many number of images need to create.')
    parser.add_argument('--suffix', type=str, default='train',
                        help='suffix for image folder and json file')
    parser.add_argument('--thread', type=int, default=5,
                        help='using how many thread to create')
    parser.add_argument('--chg_bg', type=bool, default=False,
                        help='use multiple background or not.')
    parser.add_argument('--shadow', action='store_true')
    parser.add_argument('--no-shadow', action='store_false')
    parser.add_argument('--count', type=int, default=1)  ## original=32
    parser.add_argument('--local_rank', type=int, default=0)

    args = parser.parse_args()
    ###########################################################################################
    NUM_CATEGORIES = 200
    GENERATED_NUM = args.gen_num
    ###########################################################################################
    counter = {
        'easy_mode': 0,
        'medium_mode': 0,
        'hard_mode': 0
    }
    int_2_diff = {
        1: 'easy',
        2: 'medium',
        3: 'hard'
    }
    level_dict = {}
    strategics = []
    for image_id in tqdm(range(GENERATED_NUM)):
        num_per_category, difficulty = buy_strategic(counter)
        level_dict['synthesized_image_{}'.format(image_id)] = int_2_diff[difficulty]
        strategics.append(('synthesized_image_{}'.format(image_id), num_per_category))
    # strategics_name = 'strategics_train.json'
    # if os.path.exists(strategics_name):
    #     os.remove(strategics_name)
    # with open(strategics_name, 'w') as f:
    #     json.dump(strategics, f)
    # print(counter)  # {'easy_mode': 25078, 'medium_mode': 37287, 'hard_mode': 37635}
    # quit()
    ###########################################################################################
    ###########################################################################################
    # with open(strategics_name) as f:
    #     strategics = json.load(f)
    # strategics = sorted(strategics, key=lambda s: s[0])
    version = str(GENERATED_NUM)

    output_dir = os.path.join('synthesize_{}_{}'.format(version, args.suffix))
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # if not os.path.exists(os.path.join(output_dir, 'density_maps')):
    #     os.mkdir(os.path.join(output_dir, 'density_maps'))

    mask_dir = os.path.join('synthesize_{}_{}_mask'.format(version, args.suffix))
    os.makedirs(mask_dir, exist_ok=True)

    num_threads = args.count
    sub_strategics = strategics[args.local_rank::num_threads]
    config = Config()
    DATASET_ROOT = config.get_dataset_root()
    CURRENT_ROOT = str(pathlib.Path().resolve())
    save_json_file = os.path.join(sys.path[0],
                                  'synthesize_{}_{}.json'.format(version, args.suffix))  # synthesis images json檔案
    train_json = os.path.join(DATASET_ROOT, 'retail_product_checkout',
                              'instances_train2019.json')  # rpc的原始train.json
    train_imgs_dir = os.path.join(DATASET_ROOT, 'retail_product_checkout', 'train2019')  # rpc 的train影像資料夾
    train_imgs_mask_dir = os.path.join(sys.path[0], 'extracted_masks_tracer5_morph10', 'masks')  # 擷取的mask影像資料夾
    save_mask = True

    with open('ratio_annotations.json') as fid:
        ratio_annotations = json.load(fid)
    with open('ratio_annotations_all.json') as fid:
        ratio_annotations_all = json.load(fid)
    with open('train_val_ratio.json') as fid:
        train_val_ratio = json.load(fid)
    with open(train_json) as fid:
        data = json.load(fid)
    images = {}
    for x in data['images']:
        images[x['id']] = x
    annotations = {}
    for x in data['annotations']:
        annotations[images[x['image_id']]['file_name']] = x

    # ---------------------------
    # get image list
    # ---------------------------
    object_paths = get_object_paths()
    # ---------------------------
    # items dict: Key: item_ID, value: item_images(list)
    # ---------------------------
    object_category_paths = defaultdict(list)
    for path in object_paths:
        name = os.path.basename(path)
        category = annotations[name]['category_id']
        object_category_paths[category].append(path)
    object_category_paths = dict(object_category_paths)  # store each categories all single images

    # bg_img_cv = cv2.imread('bg.jpg')
    # bg_height, bg_width = bg_img_cv.shape[:2]
    # mask_img_cv = np.zeros((bg_height, bg_width), dtype=np.uint8)

    with open('item_size_all.json') as fid:
        item_size = json.load(fid)
    json_color = {}  # store each image mask color to category
    json_ann = []
    json_img = []
    ann_idx = 0
    finished_img = 0
    m = multiprocessing.Manager()
    lock = m.Lock()
    image_left = args.gen_num
    image_cnt = 1
    MAX_JOBS_IN_QUEUE = 6
    strategics_iter = iter(strategics)
    # print(strategics)
    jobs = {}
    pbar = tqdm(total=len(strategics))
    with ThreadPoolExecutor(max_workers=MAX_JOBS_IN_QUEUE) as executor:
        # loop = tqdm(strategics_iter, total=len(strategics), leave=False)
        while image_left > 0:
            for image_id, num_per_category in strategics_iter:
                # image_id = next(iter(strat.keys()))
                # num_per_category = next(iter(strat.values()))
                # print('image_id :'.format(image_id))
                job = executor.submit(create_image, image_id, num_per_category, args.chg_bg, args.shadow,
                                      lock=lock)
                jobs[job] = image_id
                if len(jobs) > MAX_JOBS_IN_QUEUE:
                    break  # limit the job submission for now job
            for job in as_completed(jobs):
                image_cnt_res = jobs[job]
                # print("{} created".format(image_cnt_res))
                del jobs[job]
                image_left -= 1
                # ===================================
                pbar.update(1)
                # elapsed = pbar.format_dict["elapsed"]
                # rate = pbar.format_dict["rate"]
                # remaining = (pbar.total - pbar.n) / rate if rate and pbar.total else 0  # Seconds*
                # pbar.set_description("remaining %s" % remaining)
                # ===================================
                break
        print('all images created.')

    pbar.close()

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
    new_json['color'] = json_color
    if save_json_file:
        with open(save_json_file, 'w') as fid:
            json.dump(new_json, fid)
