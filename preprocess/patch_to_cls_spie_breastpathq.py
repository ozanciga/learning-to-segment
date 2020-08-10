'''
breakhis dataset patches
'''
from tqdm import tqdm
import os
import numpy as np
from myargs import args
import utils.filesystem as ufs
from PIL import Image
import csv
import glob
from utils import preprocessing

is_spie = False  # spie challenge or  wsi pipeline?
val = False

if val:
    args.patch_folder = '/home/ozan/Downloads/breastpathq/datasets/validation'
    args.label_csv_path = '/home/ozan/Downloads/breastpathq-test/val_labels.csv'
    #args.patch_folder = '/home/ozan/Downloads/breastpathq/datasets (copy)/validation'
    #args.label_csv_path = '/home/ozan/Downloads/breastpathq/datasets (copy)/val_labels.csv'
    savepath = args.val_image_pth
else:
    args.patch_folder = '/home/ozan/Downloads/breastpathq/datasets/train'
    args.label_csv_path = '/home/ozan/Downloads/breastpathq/datasets/train_labels.csv'
    #args.patch_folder = '/home/ozan/Downloads/breastpathq/datasets (copy)/train'
    #args.label_csv_path = '/home/ozan/Downloads/breastpathq/datasets (copy)/train_labels.csv'
    savepath = args.train_image_pth


if __name__ == '__main__':


    'train'
    ufs.make_folder('../' + savepath, False)
    metadata_pth_train = '../{}/gt.npy'.format(savepath)
    metadata = ufs.fetch_metadata(metadata_pth_train)

    raw_gt = {}

    cc = []

    with open('{}'.format(args.label_csv_path)) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader)
        for row in csv_reader:
            image_id = int(row[0])
            region_id = int(row[1])
            cellularity = float(row[2])
            if image_id not in raw_gt:
                raw_gt[image_id] = {}
            raw_gt[image_id][region_id] = cellularity
            cc.append(cellularity)

    cc = np.array(cc)
    cc = np.unique(cc)
    print(cc.shape, cc)

    gt = Image.fromarray(np.zeros((args.tile_h, args.tile_w), dtype=np.uint8))

    for num_images, image_path in tqdm(enumerate(glob.glob('{}/*.tif'.format(args.patch_folder)))):

        image_id, region_id = os.path.basename(image_path).split('_')
        region_id = region_id.replace('.tif', '')
        image_id, region_id = int(image_id), int(region_id)

        cellularity = raw_gt[image_id][region_id]
        if cellularity > 0:
            continue

        image = Image.open(image_path).convert('RGB')
        image = image.resize((args.tile_h, args.tile_w))

        'save paths'
        tilepth_w = '{}/w_{}_{}.png'.format(savepath, image_id, region_id)
        tilepth_g = '{}/g_{}_{}.png'.format(savepath, image_id, region_id)

        ' save images '
        image.save('../' + tilepth_w)
        gt.save('../' + tilepth_g)

        if image_id not in metadata:
            metadata[image_id] = {}

        metadata[image_id][region_id] = {
            'wsi': tilepth_w,
            'label': 0,
            'mask': tilepth_g
        }

    np.save(metadata_pth_train, metadata)
