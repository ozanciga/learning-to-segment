'''
pick 50 images from sunnybrook
10 indep cases and use them as
"normal" data samples
'''
import cv2
import openslide
import os
import numpy as np
from myargs import args
import glob
from utils.read_xml import getGT
import utils.filesystem as ufs
from tqdm import tqdm
from PIL import Image
from sklearn.cluster import KMeans
import gc
from utils.preprocessing import nextpow2

traindata_type = 'reg'  # if reg, it will generate patches with label corresponding to 0.0 (no tumors/cellularity)

args.raw_train_pth = '/home/ozan/remoteDir/'

ufs.make_folder('../' + args.train_image_pth, True)
wsipaths = glob.glob('{}/Case*/*.svs'.format(args.raw_train_pth))

' check if metadata gt.npy already exists to append to it '
metadata_pth = '../{}/gt.npy'.format(args.train_image_pth)
metadata = ufs.fetch_metadata(metadata_pth)

pwhs = {
    np.maximum(args.tile_w, args.tile_h): 0
}

annotations_list = glob.glob('{}/**/sedeen-Sherine**/*.xml'.format(args.raw_train_pth))
annotations_filename_list = [int(os.path.basename(p).replace('.session.xml', '')) for p in annotations_list]

svs_list = glob.glob('{}/Case*/*.svs'.format(args.raw_val_pth))
svs_filename_list = [(int(os.path.basename(p).replace('.svs', '')), p) for p in svs_list]
svs_ids = [p[0] for p in svs_filename_list]

benigns = [101332, 101333, 101358, 101359, 101361,
101362, 101363, 101364,101366,101372,
101376,101381,101382,101488,101492,101497,
101498,101510,99189,99190,99191,99192,
99204,99205,99206,99207,99916]

no_tumor_bed = set(svs_ids).difference(set(annotations_filename_list))
no_tumor_bed = no_tumor_bed.union(set(benigns))

no_tumor_bed = list(no_tumor_bed)
import random
random.seed(0)
random.shuffle(no_tumor_bed)
no_tumor_bed_train_ids = [101399, 100239, 101345, 101354, 101355, 100230, 101383,
101347, 101372, 101352, 101335, 100279, 101488, 100341,
101510, 101394, 100232, 101350, 99204, 101386, 100285,
101361, 100244, 99205, 100213, 100268, 100345, 100328,
101366, 101402, 101324, 100240, 100260, 101346, 100346,
99206, 101384, 100262, 101396, 101348, 101382, 100227,
101340, 101341, 101342, 99192, 100329, 101400, 100276,
101356]


wsipaths = sorted(wsipaths)
patch_id = 0

gt_patch = np.zeros((args.tile_h, args.tile_w), dtype=np.uint8)
gt_patch = Image.fromarray(gt_patch.astype(np.uint8))

for wsipath in tqdm(wsipaths):

    filename = os.path.basename(wsipath)

    svs_id = int(filename.replace('.svs', ''))

    # if this is no tumor and the ones we picked for training.
    if svs_id in no_tumor_bed_train_ids:

        'read scan and get metadata'
        scan = openslide.OpenSlide(wsipath)
        if scan.level_count < 3:
            continue

        metadata[filename] = {}

        mskpth = '../{}/{}.png'.format(args.wsi_mask_pth, filename)
        fg_mask = Image.open(mskpth).convert('L')
        fg_mask = np.array(fg_mask)

        n_labels, labels, stats, centers = cv2.connectedComponentsWithStats((fg_mask).astype(np.uint8))
        centers = centers.astype(np.int)
        '''
        stats
        [left, top, width, height, area]
        '''

        for tile_id in range(1, n_labels):

            l, u = stats[tile_id, [0, 1]]
            w, h = stats[tile_id, [2, 3]]
            area = stats[tile_id, 4]
            cx, cy = centers[tile_id, :]

            label_patch = labels[u:u + h, l:l + w] == tile_id

            l *= scan.level_downsamples[2]/scan.level_downsamples[args.scan_level]
            u *= scan.level_downsamples[2]/scan.level_downsamples[args.scan_level]
            w *= scan.level_downsamples[2]/scan.level_downsamples[args.scan_level]
            h *= scan.level_downsamples[2]/scan.level_downsamples[args.scan_level]

            cx *= scan.level_downsamples[2]/scan.level_downsamples[args.scan_level]
            cy *= scan.level_downsamples[2]/scan.level_downsamples[args.scan_level]

            if np.float(area) / (args.tile_w * args.tile_h) <= 0.01:
                continue

            pwh = nextpow2(np.maximum(w, h))
            
            if pwh <= args.scan_resize * np.maximum(args.tile_w, args.tile_h):

                pwh = args.scan_resize * np.maximum(args.tile_w, args.tile_h)
                dx = dy = pwh // 2

                if pwh not in pwhs:
                    pwhs[pwh] = 0

                up, down = np.maximum(cy-dy, 1), np.minimum(cy+dy, scan.level_dimensions[args.scan_level][1])
                left, right = np.maximum(cx-dx, 1), np.minimum(cx+dx, scan.level_dimensions[args.scan_level][0])

                if up == 1:
                    down = up + pwh
                if down == fg_mask.shape[0]:
                    up = down - pwh
                if left == 1:
                    right = left + pwh
                if right == fg_mask.shape[1]:
                    left = right - pwh

                'patch paths'
                tilepth_w = '{}/w_{}_{}.png'.format(args.train_image_pth, filename, patch_id)
                tilepth_g = '{}/g_{}_{}.png'.format(args.train_image_pth, filename, patch_id)

                ' save metadata '
                metadata[filename][patch_id] = {
                    'wsi': tilepth_w,
                    'label': tilepth_g if traindata_type == 'seg' else float(0.0),
                }

                ' save images '
                if traindata_type == 'seg':
                    gt_patch.save('../' + tilepth_g)

                wsi_patch = scan.read_region((int(left), int(up)),
                                             args.scan_level,
                                             (pwh, pwh)).convert('RGB')

                if args.scan_resize != 1:
                    wsi_patch = wsi_patch.resize((args.tile_w, args.tile_h))

                wsi_patch.save('../' + tilepth_w)

                patch_id = patch_id + 1
                pwhs[pwh] += 1

            else:

                us = 1 if fg_mask.size/area <= 0.5 else 16   # undersample region

                label_patch = Image.fromarray((255*label_patch).astype(np.uint8))
                label_patch = label_patch.resize((label_patch.size[0]//us, label_patch.size[1]//us))
                label_patch = np.asarray(label_patch)
                coords = np.transpose(np.where(label_patch))[:, ::-1]  # (x,y) pairs

                num_clusters = np.ceil((np.prod(label_patch.size)/(args.tile_w*args.tile_h))+1).astype(np.int)
                kmeans = KMeans(n_clusters=num_clusters, random_state=0)
                cnt_pts = kmeans.fit(coords).cluster_centers_  # (x,y) centers
                cnt_pts = (scan.level_downsamples[2]/scan.level_downsamples[args.scan_level] * us * cnt_pts).astype(np.int)

                for _cx, _cy in cnt_pts:

                    _cx, _cy = np.asarray([_cx, _cy])

                    _cx = _cx + l
                    _cy = _cy + u

                    pwh = args.scan_resize * np.maximum(args.tile_w, args.tile_h)
                    dx = dy = pwh // 2

                    up, down = np.maximum(_cy - dy, 1), np.minimum(_cy + dy, scan.level_dimensions[args.scan_level][1])
                    left, right = np.maximum(_cx - dx, 1), np.minimum(_cx + dx, scan.level_dimensions[args.scan_level][0])

                    if up == 1:
                        down = up + pwh
                    if down == fg_mask.shape[0]:
                        up = down - pwh
                    if left == 1:
                        right = left + pwh
                    if right == fg_mask.shape[1]:
                        left = right - pwh

                    if up >= down or left >= right:
                        continue

                    'patch paths'
                    tilepth_w = '{}/w_{}_{}.png'.format(args.train_image_pth, filename, patch_id)
                    tilepth_g = '{}/g_{}_{}.png'.format(args.train_image_pth, filename, patch_id)

                    ' save metadata '
                    metadata[filename][patch_id] = {
                        'wsi': tilepth_w,
                        'label': tilepth_g if traindata_type == 'seg' else float(0.0),
                    }

                    ' save images '
                    if traindata_type == 'seg':
                        gt_patch.save('../' + tilepth_g)

                    wsi_patch = scan.read_region((int(left), int(up)),
                                                 args.scan_level,
                                                 (pwh, pwh)).convert('RGB')

                    if args.scan_resize != 1:
                        wsi_patch = wsi_patch.resize((args.tile_w, args.tile_h))

                    wsi_patch.save('../' + tilepth_w)

                    patch_id = patch_id + 1

                    if pwh not in pwhs:
                        pwhs[pwh] = 0

                    pwhs[pwh] += 1

        del fg_mask
        del labels
        del scan


np.save(metadata_pth, metadata)

