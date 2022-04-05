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
import tracer
from tracer.TRACER import TRACER
from unet import Unet
import albumentations as albu
from albumentations.pytorch.transforms import ToTensorV2



def normPRED_np(d):
    ma = np.max(d)
    mi = np.min(d)

    dn = (d - mi) / (ma - mi)
    return dn


def do_extract(path, model_name='unet', morph_it=10):
    global net
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    # -------------------------
    # start extracting object mask, using SOD model
    # -------------------------
    tracer_cfg = tracer.getConfig()
    if net is None:
        if model_name == 'u2net':
            model_dir = 'weights/u2net_best2.pth'
            net = U2NET(in_ch=3, out_ch=1)
        elif model_name == 'unet':
            model_dir = 'weights/unet.pth'
            net = Unet(n_channels=3, n_classes=1)
        elif model_name == 'tracer0':
            model_dir = 'weights/tracer0_best.pth'
            net = tracer.TRACER(tracer_cfg)
        elif model_name == 'tracer5':
            model_dir = 'weights/tracer5_best.pth'
            net = tracer.TRACER(tracer_cfg)
        else:
            raise Exception('model not implemented.')
        net.load_state_dict(torch.load(model_dir))
        net = net.to(device)
        net.eval()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tf = None
    if model_name == 'unet' or model_name == 'u2net':
        tf = T.Compose([
            T.ToPILImage(),
            T.Resize([512, 512]),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
    elif model_name.startswith('tracer'):
        tf = T.Compose([
            T.ToPILImage(),
            T.Resize([tracer_cfg.img_size, tracer_cfg.img_size]),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
    img_tf = tf(img_rgb)
    img_tf = img_tf.unsqueeze(0)
    inputs_test = torch.tensor(img_tf, device=device, dtype=torch.float32)

    # if torch.cuda.is_available():
    #     inputs_test = Variable(img_tf.cuda())
    # else:
    #     inputs_test = Variable(img_tf)

    pred_tf = None
    with torch.no_grad():
        if model_name == 'u2net':
            pred_tf, _, _, _, _, _, _ = net(inputs_test)
        elif model_name == 'unet':
            pred_tf = net(inputs_test)
        elif model_name.startswith('tracer'):
            pred_tf, _, _ = net(inputs_test)

    pred_np = pred_tf.cpu().detach().numpy()
    pred = pred_np[:, 0, :, :]
    pred = normPRED_np(pred)
    pred = pred.squeeze()
    pred = pred * 255
    pred.astype(np.uint8)
    ret, thresh = cv2.threshold(pred, 125, 255, cv2.THRESH_BINARY)
    thresh = np.array(thresh, np.uint8) # original thresh
    # ========================================= this mask is use for paper =====================================================
    thresh_stack_origin = np.stack((thresh,) * 3, axis=-1)
    thresh_stack_origin = cv2.resize(thresh_stack_origin, (crop_x2 - crop_x1, crop_y2 - crop_y1), interpolation=cv2.INTER_AREA)
    filled_origin = cv2.cvtColor(thresh_stack_origin, cv2.COLOR_BGR2GRAY)
    filled_origin = filled_origin / 255
    # ===========================================================================================================
    ## perform morphological operation
    # rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 30))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    threshed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=morph_it)
    thresh_stack = np.stack((threshed,) * 3, axis=-1)
    ### 利用contour將多個區塊連起來
    ## ============================ this part is use for paper =============================================
    thresh_convex = threshed.copy()
    Contours, Hierarchy = cv2.findContours(thresh_convex, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in Contours:
        hull = cv2.convexHull(cnt)
        cv2.drawContours(thresh_convex, [hull], -1, color=(255, 255, 255), thickness=cv2.FILLED)
    filled_convex = cv2.resize(thresh_convex, (crop_x2 - crop_x1, crop_y2 - crop_y1), interpolation=cv2.INTER_AREA)
    filled_convex = filled_convex / 255
    ## =====================================================================================================
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

    # masked_img = origin_img * filled[:, :, None] # should use this
    masked_img = origin_img * filled_convex[:, :, None] # for paper
    compare_img = np.concatenate([origin_img, masked_img], axis=1)
    cv2.imwrite(os.path.join(compare_dir, os.path.basename(path)), compare_img)

    cv2.imwrite(os.path.join(crop_dir, os.path.basename(path)), origin_img)  # crop images
    cv2.imwrite(os.path.join(crop_mask_dir, os.path.basename(path).split('.')[0] + '.png'),
                np.array(filled * 255, dtype=np.uint8))
    # image for paper
    cv2.imwrite(os.path.join(crop_mask_origin_dir, os.path.basename(path).split('.')[0] + '.png'),
                    np.array(filled_origin * 255, dtype=np.uint8))
    cv2.imwrite(os.path.join(crop_mask_convex_dir, os.path.basename(path).split('.')[0] + '.png'),
                np.array(filled_convex * 255, dtype=np.uint8))


def extract(paths, model_name='unet', morph_it=10):
    for path in tqdm(paths):
        do_extract(path, model_name=model_name, morph_it=morph_it)


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
    config = Config()
    DATASET_ROOT = config.get_dataset_root()
    parser = ArgumentParser(description="Extract masks")
    parser.add_argument('--ann_file', type=str,
                        default=os.path.join(DATASET_ROOT, 'retail_product_checkout',
                                             'instances_train2019.json'))
    parser.add_argument('--images_dir', type=str,
                        default=os.path.join(DATASET_ROOT, 'retail_product_checkout', 'train2019'))
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

    morph_it = 10
    model_name = 'u2net'
    extract_root = 'for_paper'.format(model_name, morph_it)
    output_dir = '{}/masks'.format(extract_root)
    compare_dir = '{}/masked_images'.format(extract_root)
    crop_dir = '{}/crop_images'.format(extract_root)
    crop_mask_dir = '{}/crop_masks'.format(extract_root)
    crop_mask_origin_dir = '{}/crop_masks_origin'.format(extract_root)
    crop_mask_convex_dir = '{}/crop_masks_convex'.format(extract_root)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(compare_dir):
        os.makedirs(compare_dir)

    if not os.path.exists(crop_dir):
        os.makedirs(crop_dir)

    if not os.path.exists(crop_mask_dir):
        os.makedirs(crop_mask_dir)

    os.makedirs(crop_mask_origin_dir, exist_ok=True)
    os.makedirs(crop_mask_convex_dir, exist_ok=True)

    categories = [i + 1 for i in range(200)]

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
    net = None
    extract(paths, model_name=model_name)
