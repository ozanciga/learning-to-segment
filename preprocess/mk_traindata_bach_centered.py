'''
extract gt patches centered on patch
not random sliding windows
first generates gt mask
then goes to each object center
(connected component) and makes
patches.
if object too large, extract patches
on uniformly sampled points on the object
based on kmeans
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
from utils.preprocessing import nextpow2, find_nuclei, tile_image, isforeground, stain_normalize

istrain = True
if 0:
    args.train_image_pth = args.val_image_pth
    args.raw_train_pth = '../' + args.val_image_pth
    istrain = False

ufs.make_folder('../' + args.train_image_pth, istrain)
wsipaths = glob.glob('{}/*.svs'.format(args.raw_train_pth))

' check if metadata gt.npy already exists to append to it '
metadata_pth = '../{}/gt.npy'.format(args.train_image_pth)
metadata = ufs.fetch_metadata(metadata_pth)

pwhs = {
    np.maximum(args.tile_w, args.tile_h): 0
}
wsipaths = sorted(wsipaths)
patch_id = 0

num_labels = 1000000  # max labels
cls_hits = np.array(4 * [0])
cls_hits_total = np.array(4 * [0])

for wj, wsipath in tqdm(enumerate(wsipaths)):

    cls_hits = np.array(4 * [0])

    'read scan and get metadata'
    scan = openslide.OpenSlide(wsipath)
    filename = os.path.basename(wsipath)
    metadata[filename] = {}

    m_lvl = scan.level_downsamples[2] / scan.level_downsamples[args.scan_level]  # level multiplier

    'get actual mask, i.e. the ground truth'
    xmlpath = '{}/{}.xml'.format(args.raw_train_pth, filename.split('.svs')[0])

    gt = getGT(xmlpath, scan, level=2)

    msk_pth = '{}/{}.png'.format(args.wsi_mask_pth, filename)
    if not os.path.exists(msk_pth):
        thmb = scan.read_region((0, 0), 2, scan.level_dimensions[2]).convert('RGB')
        mask = find_nuclei(thmb, fill_mask=True)
        Image.fromarray(mask.astype(np.uint8)).save(msk_pth)
    else:
        mask = Image.open(msk_pth).convert('L')
        mask = np.asarray(mask)

    n_labels, labels, stats, centers = cv2.connectedComponentsWithStats((gt > 0).astype(np.uint8))
    centers = centers.astype(np.int)

    for tile_id in range(1, n_labels):

        l, u = stats[tile_id, [0, 1]]
        w, h = stats[tile_id, [2, 3]]
        area = stats[tile_id, 4]
        cx, cy = centers[tile_id, :]

        pwh = nextpow2(np.maximum(w, h))  # patch width/height

        if pwh <= 1 / m_lvl * args.scan_resize * np.maximum(args.tile_w, args.tile_h):

            pwh = 1 / m_lvl * args.scan_resize * np.maximum(args.tile_w, args.tile_h)
            pwh = int(pwh)
            dx = dy = pwh // 2

            if pwh not in pwhs:
                pwhs[pwh] = 0

            up, down = np.maximum(cy-dy, 1), np.minimum(cy+dy, gt.shape[0])
            left, right = np.maximum(cx-dx, 1), np.minimum(cx+dx, gt.shape[1])

            if up == 1:
                down = up + pwh
            if down == gt.shape[0]:
                up = down - pwh
            if left == 1:
                right = left + pwh
            if right == gt.shape[1]:
                left = right - pwh

            'patch paths'
            tilepth_w = '{}/w_{}_{}.png'.format(args.train_image_pth, filename, patch_id)
            tilepth_g = '{}/g_{}_{}.png'.format(args.train_image_pth, filename, patch_id)

            ' save images '
            gt_patch = gt[up:down, left:right]

            uniq = np.unique(gt_patch)
            #if len(uniq) != 2 or (len(uniq) == 2 and uniq[0] != 0):
            #    continue

            label = np.long(np.median(gt_patch[labels[up:down, left:right] == tile_id]))

            cls_hits[label] += 1

            #gt_patch = 255 * np.eye(4)[gt_patch][..., 1:]

            gt_patch = Image.fromarray(gt_patch.astype(np.uint8))
            gt_patch = gt_patch.resize((args.tile_w, args.tile_h))

            if args.scan_resize != 1:
                gt_patch = gt_patch.resize((64, 64))

            wsi_patch = scan.read_region((int(left * scan.level_downsamples[2]), int(up * scan.level_downsamples[2])),
                                         args.scan_level,
                                         (args.tile_w, args.tile_h)).convert('RGB')
            if args.scan_resize != 1:
                wsi_patch = wsi_patch.resize((64, 64))

            #wsi_patch = stain_normalize(wsi_patch)
            wsi_patch.save('../' + tilepth_w)

            ' save metadata '
            if cls_hits[label] >= num_labels:
                metadata[filename][patch_id] = {
                    'wsi': tilepth_w,
                    'label': label,
                }
            else:
                gt_patch.save('../' + tilepth_g)

                metadata[filename][patch_id] = {
                    'wsi': tilepth_w,
                    'label': label,
                    'mask': tilepth_g,
                }

            patch_id = patch_id + 1
            pwhs[pwh] += 1

        else:
            us = 1 if gt.size/area <= 0.5 else 16   # undersample region

            label_patch = labels[u:u+h, l:l+w] == tile_id

            num_clusters = np.ceil((label_patch.size/(args.tile_w*args.tile_h))+1).astype(np.int)

            label_patch = Image.fromarray((255*label_patch).astype(np.uint8))
            label_patch = label_patch.resize((label_patch.size[0]//us, label_patch.size[1]//us))
            label_patch = np.asarray(label_patch)
            coords = np.transpose(np.where(label_patch))[:, ::-1]  # (x,y) pairs

            print(filename, num_clusters)
            kmeans = KMeans(n_clusters=num_clusters, random_state=0)

            cnt_pts = kmeans.fit(coords).cluster_centers_  # (x,y) centers
            cnt_pts = (us * cnt_pts).astype(np.int)

            for _cx, _cy in cnt_pts:

                _cx, _cy = np.asarray([_cx, _cy])

                _cx = _cx + l
                _cy = _cy + u

                pwh = 1 / m_lvl * args.scan_resize * np.maximum(args.tile_w, args.tile_h)
                pwh = int(pwh)

                dx = dy = pwh // 2

                up, down = np.maximum(_cy - dy, 1), np.minimum(_cy + dy, gt.shape[0])
                left, right = np.maximum(_cx - dx, 1), np.minimum(_cx + dx, gt.shape[1])

                if up == 1:
                    down = up + pwh
                if down == gt.shape[0]:
                    up = down - pwh
                if left == 1:
                    right = left + pwh
                if right == gt.shape[1]:
                    left = right - pwh

                if up >= down or left >= right:
                    continue

                'patch paths'
                tilepth_w = '{}/w_{}_{}.png'.format(args.train_image_pth, filename, patch_id)
                tilepth_g = '{}/g_{}_{}.png'.format(args.train_image_pth, filename, patch_id)

                ' save images '
                gt_patch = gt[up:down, left:right]

                uniq = np.unique(gt_patch)
                #if len(uniq) != 2 or (len(uniq) == 2 and uniq[0] != 0):
                #    continue

                '''if (area * m_lvl ** 2) / (pwh ** 2) <= 0.25 \
                        or (area * m_lvl ** 2) / (pwh ** 2) >= 0.75 \
                        or len(np.unique(gt_patch)) != 2 \
                        or np.unique(gt_patch)[0] != 0:
                    continue'''

                label = np.long(np.median(gt_patch[labels[up:down, left:right] == tile_id]))

                cls_hits[label] += 1

                #gt_patch = 255 * np.eye(4)[gt_patch][..., 1:]

                gt_patch = Image.fromarray(gt_patch.astype(np.uint8))
                gt_patch = gt_patch.resize((args.tile_w, args.tile_h))

                if args.scan_resize != 1:
                    gt_patch = gt_patch.resize((64, 64))

                wsi_patch = scan.read_region(
                    (int(left * scan.level_downsamples[2]), int(up * scan.level_downsamples[2])),
                    args.scan_level,
                    (args.tile_w, args.tile_h)).convert('RGB')

                if args.scan_resize != 1:
                    wsi_patch = wsi_patch.resize((64, 64))

                #wsi_patch = stain_normalize(wsi_patch)
                wsi_patch.save('../' + tilepth_w)

                ' save metadata '
                if cls_hits[label] >= num_labels:
                    metadata[filename][patch_id] = {
                        'wsi': tilepth_w,
                        'label': label,
                    }
                else:
                    gt_patch.save('../' + tilepth_g)

                    metadata[filename][patch_id] = {
                        'wsi': tilepth_w,
                        'label': label,
                        'mask': tilepth_g,
                    }

                patch_id = patch_id + 1

                if pwh not in pwhs:
                    pwhs[pwh] = 0

                pwhs[pwh] += 1

    cls_hits_total += cls_hits

    del gt
    del labels
    del scan

print(cls_hits_total)
np.save(metadata_pth, metadata)

print(pwhs)
