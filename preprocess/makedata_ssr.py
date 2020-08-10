'''
same size regions (ssr)
option 1: extract region
w/ gt segmentation mask
option 2: extract region
in a bounding box,
and with classification
label
'''
from utils import preprocessing
from PIL import Image
import numpy as np
import openslide
import cv2
from myargs import args
import os
from utils.read_xml import getGT
import glob
from tqdm import tqdm
import shutil
from operator import itemgetter
from scipy.stats import mode
import utils.filesystem as ufs


option = 'classification'

dx = 0
dy = 0

savefolder = [
    '../data/ssr/train',
    '../data/ssr/val',
]

indices = [
    [2, 3, 4, 5, 6, 7, 9],
    [0, 1, 8],
]
args.raw_train_pth = '/home/ozan/ICIAR2018_BACH_Challenge/WSI/'
wsipaths = sorted(glob.glob('{}/A*.svs'.format(args.raw_train_pth)))

for ij in range(2):

    if os.path.exists(savefolder[ij]):
        shutil.rmtree(savefolder[ij])
    os.makedirs(savefolder[ij])

    if option == 'classification':
        ' check if metadata gt.npy already exists to append to it '
        metadata_pth = '{}/gt.npy'.format(savefolder[ij])
        metadata_pth = ufs.fix_path(metadata_pth)
        metadata = ufs.fetch_metadata(metadata_pth)

    region_id = 0

    for wsipath in itemgetter(*indices[ij])(wsipaths):

        scan = openslide.OpenSlide(wsipath)
        filename = os.path.basename(wsipath)
        'get actual mask, i.e. the ground truth'
        xmlpath = '{}/{}.xml'.format(args.raw_train_pth, filename.split('.svs')[0])
        gt = getGT(xmlpath, scan, level=args.scan_level)
        gt_rgb = np.eye(4)[gt][..., 1:]

        x_max, y_max = scan.level_dimensions[args.scan_level]

        n_labels, labels, stats, centers = cv2.connectedComponentsWithStats((gt > 0).astype(np.uint8))
        centers = centers.astype(np.int)

        for tile_id in tqdm(range(1, n_labels), disable=True):

            l, u = stats[tile_id, [0, 1]]
            w, h = stats[tile_id, [2, 3]]
            area = stats[tile_id, 4]
            cx, cy = centers[tile_id, :]

            l_ = np.maximum(l - dx, 1)
            u_ = np.maximum(u - dy, 1)

            r_ = np.minimum(l + w + 2 * dx, scan.level_dimensions[args.scan_level][0])
            d_ = np.minimum(u + h + 2 * dy, scan.level_dimensions[args.scan_level][1])

            w_ = r_ - l_
            h_ = d_ - u_

            if w_ * h_ < 2**29:

                savepath = '{}/{}_image.png'.format(savefolder[ij], region_id)

                if option == 'segmentation':

                    region = scan.read_region((l_ * (4 ** args.scan_level), u_ * (4 ** args.scan_level)), args.scan_level, (w_, h_)).convert('RGB')
                    region = region.resize((args.tile_w, args.tile_h))

                    gt_region = gt_rgb[u_:u_ + h_, l_:l_ + w_, ...]
                    gt_region = Image.fromarray(255*gt_region.astype(np.uint8))\
                        .resize((args.tile_w, args.tile_h))
                    gt_region.save('{}/{}_gt.png'.format(savefolder[ij], region_id))

                elif option == 'classification':

                    times = 1
                    '''
                    times = (w_*h_/(args.tile_w*args.tile_h))
                    times *= 7
                    times = np.minimum(times, 7)
                    times = np.floor(times)
                    times = np.maximum(1, times).astype(np.int)

                    # center coordinate of region
                    c_x = int(0.5*(r_ + l_))
                    c_y = int(0.5*(d_ + u_))

                    l_, r_ = c_x - args.tile_w // 2, c_x + args.tile_w // 2
                    u_, d_ = c_y - args.tile_h // 2, c_y + args.tile_h // 2

                    w_ = r_ - l
                    h_ = d_ - u_
                    '''

                    region = scan.read_region((l_ * (4 ** args.scan_level), u_ * (4 ** args.scan_level)), args.scan_level, (w_, h_)).convert('RGB')
                    region = region.resize((args.tile_w, args.tile_h))

                    current_label = mode(gt[labels == tile_id])[0][0]

                    filename = os.path.basename(wsipath)
                    if filename not in metadata:
                        metadata[filename] = {}

                    metadata[filename][tile_id] = {
                        'image': savepath,
                        'label': current_label,
                        'times': times
                    }

                region.save(savepath)

                region_id += 1

    if option == 'classification':
        np.save(metadata_pth, metadata)
