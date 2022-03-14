from PIL import Image  # (pip install Pillow)
import numpy as np  # (pip install numpy)
from skimage import measure  # (pip install scikit-image)
from shapely.geometry import Polygon, MultiPolygon  # (pip install Shapely)
import time
import matplotlib.pyplot as plt
import glob
import os
from tqdm import tqdm
import multiprocessing
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

"""
create annotation for rpc dataset 

after running this program create segmentation data
please run the utils.py check_coco_seg_format() to make sure json data is correct 
"""


def create_sub_masks(mask_image):
    width, height = mask_image.size

    # Initialize a dictionary of sub-masks indexed by RGB colors
    sub_masks = {}
    for x in range(width):
        for y in range(height):
            # Get the RGB values of the pixel
            pixel = mask_image.getpixel((x, y))[:3]

            # If the pixel is not black...
            if pixel != (0, 0, 0):
                # Check to see if we've created a sub-mask...
                pixel_str = str(pixel)
                sub_mask = sub_masks.get(pixel_str)
                if sub_mask is None:
                    # Create a sub-mask (one bit per pixel) and add to the dictionary
                    # Note: we add 1 pixel of padding in each direction
                    # because the contours module doesn't handle cases
                    # where pixels bleed to the edge of the image
                    sub_masks[pixel_str] = Image.new('1', (width + 2, height + 2))

                # Set the pixel value to 1 (default is 0), accounting for padding
                sub_masks[pixel_str].putpixel((x + 1, y + 1), 1)

    return sub_masks


def create_sub_mask_annotation(sub_mask: np.ndarray, image_id, category_id, annotation_id, is_crowd, lock: Lock):
    """
    # Find contours (boundary lines) around each sub-mask
    # Note: there could be multiple contours if the object
    # is partially occluded. (E.g. an elephant behind a tree)
    Args:
        sub_mask:np.ndarray
        image_id:
        category_id:
        annotation_id:
        is_crowd:

    Returns:

    """

    contours = measure.find_contours(sub_mask, 0.5, positive_orientation='low')

    segmentations = []
    polygons = []
    for contour in contours:
        # Flip from (row, col) representation to (x, y)
        # and subtract the padding pixel
        for i in range(len(contour)):
            row, col = contour[i]
            contour[i] = (col - 1, row - 1)

        # Make a polygon and simplify it
        poly = Polygon(contour)
        poly = poly.simplify(1.0, preserve_topology=False)
        if type(poly) is Polygon:
            polygons.append(poly)
            segmentation = np.array(poly.exterior.coords).ravel().tolist()
            segmentations.append(segmentation)
        elif type(poly) is MultiPolygon:
            for p in poly:
                polygons.append(p)
                segmentation = np.array(p.exterior.coords).ravel().tolist()
                segmentations.append(segmentation)
        else:
            raise Exception("poly got type: {}".format(type(poly)))

    # Combine the polygons to calculate the bounding box and area
    multi_poly = MultiPolygon(polygons)
    x, y, max_x, max_y = multi_poly.bounds
    width = max_x - x
    height = max_y - y
    bbox = (x, y, width, height)
    area = multi_poly.area

    annotation = {
        'segmentation': segmentations,
        'iscrowd': is_crowd,
        'image_id': image_id,
        'category_id': category_id,
        'id': annotation_id,
        'bbox': bbox,
        'area': area
    }

    return annotation


if __name__ == "__main__":
    print("creating annotations")
    import json

    # plant_book_mask_image = Image.open('./example_images/plant_book_mask.png').convert('RGB')
    # bottle_book_mask_image = Image.open('./example_images/bottle_book_mask.png').convert('RGB')
    #
    # mask_images = [plant_book_mask_image, bottle_book_mask_image]
    name = 'synthesize_30000_mix'
    mask_path = '/media/ian/WD/datasets/rpc_tmp/{}_mask_small'.format(name)
    json_path = '/media/ian/WD/datasets/rpc_tmp/{}_small.json'.format(name)
    mask_images = glob.glob(os.path.join(mask_path, "*.png"))
    with open(json_path) as fid:
        json_data = json.load(fid)
    color_infos = json_data['color']
    imgs = json_data['images']
    anns = json_data['annotations']
    anns = sorted(anns, key=lambda i: (i['image_id'], i['id']))
    filename_2_imgId = {img['file_name'].split('.')[0]: img['id'] for img in imgs}
    # Define which colors match which categories in the images
    # houseplant_id, book_id, bottle_id, lamp_id = [1, 2, 3, 4]
    # category_ids = {
    #     1: {
    #         '(0, 255, 0)': houseplant_id,
    #         '(0, 0, 255)': book_id,
    #     },
    #     2: {
    #         '(255, 255, 0)': bottle_id,
    #         '(255, 0, 128)': book_id,
    #         '(255, 100, 0)': lamp_id,
    #     }
    # }

    is_crowd = 0

    # These ids will be automatically increased as we go
    annotation_id = 1
    # image_id = 1

    # Create the annotations
    annotations = []
    ann_left = len(anns)
    mask_images_iter = iter(mask_images)
    pbar = tqdm(total=len(anns))
    # ======== multi-thread config ============= #
    m = multiprocessing.Manager()
    lock = m.Lock()
    jobs = {}
    MAX_JOBS_IN_QUEUE = 20
    # ========================================== #
    with ProcessPoolExecutor() as executor:
        while ann_left > 0:
            mask_image = next(mask_images_iter)
            filename = os.path.basename(mask_image)
            mask_image = Image.open(mask_image).convert('RGB')
            sub_masks = create_sub_masks(mask_image)
            image_id = filename_2_imgId[filename.split('.')[0]]
            category_ids = color_infos[filename.split('.')[0]]['color_dict']
            category_ids_iter = iter(category_ids.items())
            color_left = len(category_ids.keys())
            while color_left > 0:
                for color, cat in category_ids_iter:
                    try:
                        sub_mask = sub_masks[color]
                        # category_id = category_ids[image_id][color]
                        ## show sub_mask
                        # plt.imshow(sub_mask)
                        # plt.show()
                        sub_mask_np = np.array(sub_mask)
                        # annotation = create_sub_mask_annotation(sub_mask=sub_mask_np,
                        #                                         image_id=image_id,
                        #                                         category_id=cat,
                        #                                         annotation_id=annotation_id,
                        #                                         is_crowd=is_crowd)
                        job = executor.submit(create_sub_mask_annotation, sub_mask_np, image_id, cat, annotation_id,
                                              is_crowd, lock=lock)
                        jobs[job] = annotation_id
                        annotation_id += 1
                        if len(jobs) > MAX_JOBS_IN_QUEUE:
                            break  # limit the job submission for now job
                        # annotations.append(annotation)

                    except KeyError as ke:
                        print("filename: {}".format(filename))
                        print("sub_mask got: {}".format(sub_masks))
                        print("got KeyError with color: {}".format(color))
                        color_left -= 1
                        ann_left -= 1
                        break
                for job in as_completed(jobs):
                    # ann_cnt_res = jobs[job]
                    # print("ann: {} created".format(ann_cnt_res))
                    annotation = job.result()
                    annotations.append(annotation)
                    del jobs[job]
                    ann_left -= 1
                    color_left -= 1
                    pbar.update(1)
                    break
    pbar.close()
    json_data['annotations'] = annotations
    del json_data['color']
    with open('/media/ian/WD/datasets/rpc_tmp/{}_small_seg.json'.format(name), 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)
# print(json.dumps(annotations))
