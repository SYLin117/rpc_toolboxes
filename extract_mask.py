import glob
import json
import os
import sys
from argparse import ArgumentParser

import cv2
import numpy as np
from scipy import ndimage
from tqdm import tqdm

import torch
import torchvision.transforms as T
from torch.autograd import Variable

from config import Config

from u2net_model import U2NET, U2NETP
from unet import Unet


def normPRED_np(d):
    ma = np.max(d)
    mi = np.min(d)

    dn = (d - mi) / (ma - mi)
    return dn


def do_extract(path):
    annotation = annotations[os.path.basename(path)]
    bbox = annotation['bbox']
    x, y, w, h = [int(x) for x in bbox]
    img = cv2.imread(path)
    origin_height, origin_width = img.shape[:2]

    box_pad = 5
    crop_x1 = x - box_pad if (x - box_pad) >= 0 else 0
    crop_y1 = y - box_pad if (y - box_pad) >= 0 else 0
    crop_x2 = x + w + box_pad if (x + w + box_pad) <= origin_width else origin_width
    crop_y2 = y + h + box_pad if (y + h + box_pad) <= origin_height else origin_height

    x = x - crop_x1
    y = y - crop_y1

    # origin_img = img = img[crop_y1:crop_y2, crop_x1:crop_x2]
    img = img[crop_y1:crop_y2, crop_x1:crop_x2]
    origin_img = np.copy(img)

    # # --------------------------
    # # start extracting object mask(original steps)
    # # --------------------------
    # img = cv2.bilateralFilter(img, 3, 75, 75)
    #
    # # -------------------------
    # # edge detect
    # # -------------------------
    # ## version 1
    # edges = detector.detectEdges(np.float32(img) / 255)
    # ## version 2
    # # blob = cv2.dnn.blobFromImage(img, scalefactor=1.0, size=(img.shape[1], img.shape[0]),
    # #                              mean=(104.00698793, 116.66876762, 122.67891434),
    # #                              swapRB=False, crop=False)
    # # net.setInput(blob)
    # # hed = net.forward()
    # # edges = cv2.resize(hed[0, 0], (img.shape[1], img.shape[0]))
    # # -------------------------
    # # edge process
    # # -------------------------
    # object_box_mask = np.zeros_like(edges, dtype=np.uint8)
    # object_box_mask[y:y + h, x:x + w] = 1
    # edges[(1 - object_box_mask) == 1] = 0
    # edges[(edges < (edges.mean() * 0.5)) & (edges < 0.1)] = 0
    #
    # # -------------------------
    # # erode and dilate
    # # -------------------------
    # filled = ndimage.binary_fill_holes(edges).astype(np.uint8)
    # filled = cv2.erode(filled, np.ones((32, 32), np.uint8))
    # filled = cv2.dilate(filled, np.ones((32, 32), np.uint8))
    # filled = cv2.erode(filled, np.ones((8, 8), np.uint8))
    # filled = cv2.medianBlur(filled, 17)  # fill is annotation's bbox area
    # # -------------------------
    # # end of mask extraction(original steps)
    # # -------------------------

    # -------------------------
    # start extracting object mask, using SOD model
    # -------------------------
    model_name = 'unet'
    if model_name == 'u2net':
        model_dir = 'weights/u2net_best.pth'
        net = U2NET(in_ch=3, out_ch=1)
    elif model_name == 'unet':
        model_dir = 'weights/unet.pth'
        net = Unet(n_channels=3, n_classes=1)
    else:
        raise Exception('model not implemented.')
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tf = T.Compose([
        T.ToPILImage(),
        T.Resize([512, 512]),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    img_tf = tf(img_rgb)
    img_tf = img_tf.unsqueeze(0)
    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_dir))
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_dir, map_location='cpu'))

    if torch.cuda.is_available():
        inputs_test = Variable(img_tf.cuda())
    else:
        inputs_test = Variable(img_tf)

    pred_tf = None
    if model_name == 'u2net':
        pred_tf, _, _, _, _, _, _ = net(inputs_test)
    elif model_name == 'unet':
        pred_tf = net(inputs_test)

    pred_np = pred_tf.cpu().detach().numpy()
    pred = pred_np[:, 0, :, :]
    pred = normPRED_np(pred)
    pred = pred.squeeze()
    pred = pred * 255
    pred.astype(np.uint8)
    ret, thresh = cv2.threshold(pred, 125, 255, cv2.THRESH_BINARY)
    thresh = np.array(thresh, np.uint8)
    ## perform morphological operation
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 30))
    threshed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, rect_kernel)
    thresh_stack = np.stack((thresh,) * 3, axis=-1)
    ## 利用contour將多個區塊連起來
    # Contours, Hierarchy = cv2.findContours(threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # for cnt in Contours:
    #     hull = cv2.convexHull(cnt)
    #     cv2.drawContours(thresh_stack, [hull], -1, color=(255, 255, 255), thickness=cv2.FILLED)
    thresh_stack = cv2.resize(thresh_stack, (crop_x2 - crop_x1, crop_y2 - crop_y1), interpolation=cv2.INTER_AREA)
    filled = cv2.cvtColor(thresh_stack, cv2.COLOR_BGR2GRAY)
    filled = filled / 255
    # cv2.imshow('filled', filled)
    # cv2.waitKey()
    # cv2.destroyWindow('filled')
    # -------------------------
    # end of object extraction
    # -------------------------

    save_image = np.zeros((origin_height, origin_width), np.uint8)  # save_image is full size image's mask
    save_image[crop_y1:crop_y2, crop_x1:crop_x2] = np.array(filled * 255, dtype=np.uint8)
    # cv2.imshow('save_image', save_image)
    # cv2.waitKey()
    # cv2.destroyWindow('save_image')
    cv2.imwrite(os.path.join(output_dir, os.path.basename(path).split('.')[0] + '.png'),
                save_image)  ## original image size mask

    masked_img = origin_img * filled[:, :, None]
    compare_img = np.concatenate([origin_img, masked_img], axis=1)
    cv2.imwrite(os.path.join(compare_dir, os.path.basename(path)), compare_img)

    # cv2.imwrite(os.path.join(crop_dir, os.path.basename(path)), origin_img)  # crop images
    # cv2.imwrite(os.path.join(crop_mask_dir, os.path.basename(path).split('.')[0] + '.png'),
    #             np.array(filled * 255, dtype=np.uint8))


def extract(paths):
    for path in tqdm(paths):
        do_extract(path)


class CropLayer(object):
    def __init__(self, params, blobs):
        # initialize our starting and ending (x, y)-coordinates of
        # the crop
        self.startX = 0
        self.startY = 0
        self.endX = 0
        self.endY = 0

    def getMemoryShapes(self, inputs):
        # the crop layer will receive two inputs -- we need to crop
        # the first input blob to match the shape of the second one,
        # keeping the batch size and number of channels
        (inputShape, targetShape) = (inputs[0], inputs[1])
        (batchSize, numChannels) = (inputShape[0], inputShape[1])
        (H, W) = (targetShape[2], targetShape[3])
        # compute the starting and ending crop coordinates
        self.startX = int((inputShape[3] - targetShape[3]) / 2)
        self.startY = int((inputShape[2] - targetShape[2]) / 2)
        self.endX = self.startX + W
        self.endY = self.startY + H
        # return the shape of the volume (we'll perform the actual
        # crop during the forward pass
        return [[batchSize, numChannels, H, W]]

    def forward(self, inputs):
        # use the derived (x, y)-coordinates to perform the crop
        return [inputs[0][:, :, self.startY:self.endY,
                self.startX:self.endX]]


if __name__ == '__main__':
    parser = ArgumentParser(description="Extract masks")
    parser.add_argument('--ann_file', type=str,
                        default='D:\\datasets\\retail_product_checkout\\instances_train2019.json')
    parser.add_argument('--images_dir', type=str,
                        default='D:\\datasets\\retail_product_checkout\\train2019')
    parser.add_argument('--model_file', type=str, default='model.yml.gz')
    parser.add_argument('--caffemodel', type=str, default='hed_pretrained_bsds.caffemodel')
    parser.add_argument('--prototxt', type=str, default='deploy.prototxt')
    args = parser.parse_args()

    with open(args.ann_file) as fid:
        data = json.load(fid)
    images = {}
    for x in data['images']:
        images[x['id']] = x
    annotations = {}
    for x in data['annotations']:
        annotations[images[x['image_id']]['file_name']] = x

    extract_root = 'extracted_masks'
    output_dir = '{}/masks'.format(extract_root)
    compare_dir = '{}/masked_images'.format(extract_root)
    crop_dir = '{}/crop_images'.format(extract_root)
    crop_mask_dir = '{}/crop_masks'.format(extract_root)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(compare_dir):
        os.makedirs(compare_dir)

    if not os.path.exists(crop_dir):
        os.makedirs(crop_dir)

    if not os.path.exists(crop_mask_dir):
        os.makedirs(crop_mask_dir)

    categories = [i + 1 for i in range(200)]
    config = Config()
    paths = list()
    if not config.img_filters:
        paths = glob.glob(os.path.join(args.images_dir, '*.jpg'))
    else:
        for img_filter in config.img_filters:
            paths += glob.glob(os.path.join(args.images_dir, img_filter))
    ## version 1
    # detector = cv2.ximgproc.createStructuredEdgeDetection(args.model_file)
    ## version 2
    # net = cv2.dnn.readNetFromCaffe(args.prototxt, args.caffemodel)
    # cv2.dnn_registerLayer("Crop", CropLayer)
    extract(paths)
