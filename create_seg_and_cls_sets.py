import numpy as np
from PIL import Image
import glob
from tqdm import tqdm
from mahotas import bwperim
from collections import namedtuple
import os
import shutil
import utils.filesystem as ufs
from myargs import args

mode = 'val'

dataset_name = f'digestpath2019/{mode}'

if os.path.exists(f'{dataset_name}/'):
    shutil.rmtree(f'{dataset_name}/')
os.makedirs(f'{dataset_name}/')

Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')

def area(a, b):  # returns None if rectangles don't intersect
    dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
    dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
    if (dx>=0) and (dy>=0):
        # print(dx*dy/ps**2)
        return dx*dy
    return 0


ps = 224
thresh = 1 - 0.8

paths = glob.glob(f'/home/osha/Desktop/ld3/validation_utils/data/segmentation/digestpath2019/{mode}/*_mask.png')

metadata_pth = '{}/gt.npy'.format(dataset_name)
metadata = ufs.fetch_metadata(metadata_pth)

for path in tqdm(paths, disable=0):

    metadata[os.path.basename(path)] = {}

    index = 0

    image_path = path.replace('_mask.png', '.png')

    if not os.path.exists(image_path):
        continue

    image = np.array(Image.open(image_path))

    mask = Image.open(path)

    mask = np.array(mask)

    if mode == 'train':
        for y in range(0, mask.shape[0]-ps, ps):
            for x in range(0, mask.shape[1]-ps, ps):

                tilepth_w = f'{dataset_name}/w_{os.path.basename(path)}_{index}.png'

                ymin, ymax, xmin, xmax = y, y+ps, x, x+ps

                cls_code = np.bincount(mask[ymin:ymax, xmin:xmax].flatten(), minlength=args.num_classes)
                argsorted = np.argsort(cls_code)
                cls_code_ = argsorted[-1]
                '''if np.argmax(cls_code) == 0:
                    if cls_code[argsorted[-1]] <= 3 * cls_code[argsorted[-2]]:
                        cls_code_ = argsorted[-2]

                if cls_code_ == 0:
                    continue'''

                Image.fromarray(
                    np.uint8(image[ymin:ymax, xmin:xmax])
                ).save(tilepth_w)

                metadata[os.path.basename(path)][index] = {
                    'wsi': tilepth_w,
                    'label': int(cls_code_),
                    'mask': None,
                }
                index += 1

    mask_edge = np.zeros_like(mask)

    cls = np.unique(mask)

    rect_list = []

    if len(cls) > 1:
        for cls_id in cls:
            mask_edge = np.bitwise_or(mask_edge, np.uint8(bwperim(mask == cls_id)))

    y, x = np.where(mask_edge)

    # n_class = mask.max() + 1

    for j in range(len(y)):
        center_x, center_y = x[j], y[j]
        center_x = np.maximum(ps // 2, center_x)
        center_x = np.minimum(mask.shape[1] - ps // 2, center_x)
        center_y = np.maximum(ps // 2, center_y)
        center_y = np.minimum(mask.shape[0] - ps // 2, center_y)

        ymin, ymax, xmin, xmax = center_y-ps//2, center_y+ps//2, center_x-ps//2, center_x+ps//2

        ra = Rectangle(xmin, ymin, xmax, ymax)

        if len(rect_list) < 1:
            rect_list.append(ra)
            ints = False
        else:
            ints = False
            for rb in rect_list:
                if area(ra, rb) / (ps ** 2) >= thresh:
                    ints = True
                    break
        if not ints:

            rect_list.append(ra)

            tilepth_w = f'{dataset_name}/w_{os.path.basename(path)}_{index}.png'
            tilepth_m = f'{dataset_name}/m_{os.path.basename(path)}_{index}.png'
            tilepth_g = f'{dataset_name}/g_{os.path.basename(path)}_{index}.png'

            cls_code = np.bincount(mask[ymin:ymax, xmin:xmax].flatten(), minlength=args.num_classes)
            argsorted = np.argsort(cls_code)
            cls_code_ = argsorted[-1]
            if np.argmax(cls_code) == 0:
                if cls_code[argsorted[-1]] <= 3 * cls_code[argsorted[-2]]:
                    cls_code_ = argsorted[-2]

            # gt = cls_code * np.ones((args.tile_h, args.tile_w), dtype=np.uint8)

            Image.fromarray(
                np.uint8(1 * mask[ymin:ymax, xmin:xmax])
            ).save(tilepth_m)
            Image.fromarray(
                np.uint8(image[ymin:ymax, xmin:xmax])
            ).save(tilepth_w)

            metadata[os.path.basename(path)][index] = {
                'wsi': tilepth_w,
                'label': int(cls_code_),
                'mask': tilepth_m,
            }

            index += 1

np.save('{}/gt.npy'.format(dataset_name), metadata)
