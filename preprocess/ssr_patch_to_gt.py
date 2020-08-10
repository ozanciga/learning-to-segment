'''
same size regions
patches to gt
'''
from tqdm import tqdm
import os
import numpy as np
from myargs import args
import glob
import utils.preprocessing as preprocessing
import utils.filesystem as ufs
from PIL import Image
from pathlib import Path

option = 'classification'

args.patch_folder = '/home/ozan/ICIAR2018_BACH_Challenge/Photos'
args.train_image_pth = '../data/ssr/train'

base_path = Path(__file__).parent
args.train_image_pth = (base_path / args.train_image_pth).resolve().as_posix()

if option == 'classification':

    ' check if metadata gt.npy already exists to append to it '
    metadata_pth = '{}/gt.npy'.format(args.train_image_pth)
    metadata_pth = ufs.fix_path(metadata_pth)
    metadata = ufs.fetch_metadata(metadata_pth)


if __name__ == '__main__':

    ' map class names to codes '
    cls_codes = {
        'Normal': 0,
        'Benign': 1,
        'InSitu': 2,
        'Invasive': 3
    }

    cls_folders = glob.glob('{}/*/'.format(args.patch_folder))

    num_tiles = 0

    for cls_folder in tqdm(cls_folders):
        cls_name = cls_folder.split('/')[-2]
        cls_code = cls_codes[cls_name]

        'gt will be a single number over the patch since preds are on patch level (not pixel) '
        gt = np.zeros((args.tile_h, args.tile_w, 3), dtype=np.uint8)
        if cls_code > 0:
            gt[..., cls_code - 1] = 255
        gt = Image.fromarray(gt)

        image_paths = sorted(glob.glob('{}*.png'.format(cls_folder)))
        for image_path in image_paths:

            filename = os.path.basename(image_path)
            image = Image.open(image_path).convert('RGB')

            x, y = args.tile_w, args.tile_h
            image = image.resize((x, y))

            'set up paths'
            tilepth_w = '{}/{}_image.png'.format(args.train_image_pth, filename)
            tilepth_g = '{}/{}_gt.png'.format(args.train_image_pth, filename)

            ' save images '
            image.save(tilepth_w)
            if option == 'segmentation':
                gt.save(tilepth_g)
            elif option == 'classification':
                metadata[filename] = {}
                metadata[filename][0] = {
                    'image': tilepth_w,
                    'label': cls_code,
                    'times': 7,
                }

    if option == 'classification':
        np.save(metadata_pth, metadata)
