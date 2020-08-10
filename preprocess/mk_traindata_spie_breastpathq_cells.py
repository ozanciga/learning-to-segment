'''
use cell images (dots)
to perform segmentation
'''
from tqdm import tqdm
import os
import numpy as np
from myargs import args
import utils.filesystem as ufs
from PIL import Image
import csv
import glob
import cv2
from utils import preprocessing

args.patch_folder = '/home/ozan/Downloads/breastpathq/datasets/cells'

if __name__ == '__main__':

    'train'
    ufs.make_folder('../' + args.train_image_pth, False)
    metadata_pth_train = '../{}/gt.npy'.format(args.train_image_pth)
    metadata = ufs.fetch_metadata(metadata_pth_train)

    for num_images, image_path in tqdm(enumerate(glob.glob('{}/*_crop.tif'.format(args.patch_folder)))):

        filename = os.path.basename(image_path)
        metadata[filename] = {}

        image = Image.open(image_path).convert('RGB')
        image = image.resize((args.tile_h, args.tile_w))
        image = preprocessing.quantize_image(image)

        gt_path = image_path.replace('_crop', '_mask')
        #gt_path = '/home/ozan/Downloads/breastpathq/datasets/cells/1_Region 1_mask.tif'
        gt = Image.open(gt_path).convert('RGB')
        gt = np.asarray(gt)
        str_elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        gt = cv2.dilate((np.asarray(gt) < 1).astype(np.uint8), str_elem, iterations=1)
        # gt = (np.asarray(gt) < 1).astype(np.uint8)
        gt = 1 * (gt.sum(-1) > 0).astype(np.uint8)
        gt = Image.fromarray(gt).convert('L')
        #gt = Image.fromarray(gt*255)
        gt = gt.resize((args.tile_h, args.tile_w))
        #gt = gt.resize((224, 224))
        #gt.save('x.png')

        'save paths'
        tilepth_w = '{}/w_{}_{}.png'.format(args.train_image_pth, filename, 0)
        tilepth_g = '{}/g_{}_{}.png'.format(args.train_image_pth, filename, 0)

        tilepth_w = tilepth_w.replace(' ', '_')
        tilepth_g = tilepth_g.replace(' ', '_')

        ' save metadata '
        metadata[filename][0] = {
            'wsi': tilepth_w,
            'label': tilepth_g,
        }

        ' save images '
        image.save('../' + tilepth_w)
        gt.save('../' + tilepth_g)

    np.save(metadata_pth_train, metadata)
