from pycocotools.coco import COCO

import matplotlib.pyplot as plt
import cv2

import os
import numpy as np
import random

"""
將自己標記的ann轉換成mask作為SOD模型的training資料
"""

cocoRoot = 'D:\\datasets\\retail_product_checkout\\crop_subset'

dataType = "train2017"
annFIle = os.path.join(cocoRoot, f'annotations\\instances_{dataType}.json')
mask_save_folder = os.path.join(cocoRoot, 'mask', )

coco = COCO(annFIle)
ids = coco.getCatIds('product')[0]
cats = coco.loadCats(ids)
imgIds = coco.getImgIds(catIds=[2])
if not os.path.exists(mask_save_folder):
    os.makedirs(mask_save_folder)
for imgId in imgIds:
    imgInfo = coco.loadImgs(imgId)[0]
    imPath = os.path.join(cocoRoot, 'images', dataType, imgInfo['file_name'])
    im = cv2.imread(imPath)
    annIds = coco.getAnnIds(imgIds=imgInfo['id'])
    anns = coco.loadAnns(annIds)
    mask = coco.annToMask(anns[0])
    cv2.imwrite(os.path.join(cocoRoot, 'mask', os.path.basename(imgInfo['file_name']) + ".png"), mask * 255)
