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
from utils.read_xml_sunnybrook import getGT
import utils.filesystem as ufs
from tqdm import tqdm
from PIL import Image
from sklearn.cluster import KMeans
import gc
from utils.preprocessing import nextpow2

args.raw_train_pth = 'data/sunnybrook/WSI'

ufs.make_folder('../' + args.train_image_pth, True)
wsipaths = glob.glob('../{}/*.svs'.format(args.raw_train_pth))

' check if metadata gt.npy already exists to append to it '
metadata_pth = '../{}/gt.npy'.format(args.train_image_pth)
metadata = ufs.fetch_metadata(metadata_pth)

pwhs = {
    np.maximum(args.tile_w, args.tile_h): 0
}
wsipaths = sorted(wsipaths)
patch_id = 0

num_iters = 1  # each iter randomizes the centers of objects

for _ in range(num_iters):

    for wsipath in tqdm(wsipaths):

        'read scan and get metadata'
        scan = openslide.OpenSlide(wsipath)
        filename = os.path.basename(wsipath)
        metadata[filename] = {}

        'get actual mask, i.e. the ground truth'
        xmlpath = '../{}/{}.xml'.format(args.raw_train_pth, filename.split('.svs')[0])

        gt = getGT(xmlpath, scan, level=args.scan_level)

        n_labels, labels, stats, centers = cv2.connectedComponentsWithStats((gt > 0).astype(np.uint8))
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

            pwh = nextpow2(np.maximum(w, h))  # patch width/height

            if pwh < 16:
                continue

            if pwh <= args.scan_resize * np.maximum(args.tile_w, args.tile_h):

                pwh = args.scan_resize * np.maximum(args.tile_w, args.tile_h)
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

                ' save metadata '
                metadata[filename][patch_id] = {
                    'wsi': tilepth_w,
                    'label': tilepth_g,
                }

                ' save images '
                gt_patch = gt[up:down, left:right]
                # gt_patch = (255 * np.eye(args.num_classes)[gt_patch][..., 1:]).astype(np.uint8)
                gt_patch = Image.fromarray(gt_patch.astype(np.uint8))

                if args.scan_resize != 1:
                    gt_patch = gt_patch.resize((args.tile_w, args.tile_h))

                gt_patch.save('../' + tilepth_g)

                wsi_patch = scan.read_region((int(left * scan.level_downsamples[args.scan_level]), int(up * scan.level_downsamples[args.scan_level])),
                                             args.scan_level,
                                             (pwh, pwh)).convert('RGB')
                if args.scan_resize != 1:
                    wsi_patch = wsi_patch.resize((args.tile_w, args.tile_h))

                wsi_patch.save('../' + tilepth_w)

                patch_id = patch_id + 1
                pwhs[pwh] += 1

            else:

                us = 1 if gt.size/area <= 0.5 else 16   # undersample region

                label_patch = labels[u:u+h, l:l+w] == tile_id
                label_patch = Image.fromarray((255*label_patch).astype(np.uint8))
                label_patch = label_patch.resize((label_patch.size[0]//us, label_patch.size[1]//us))
                label_patch = np.asarray(label_patch)
                coords = np.transpose(np.where(label_patch))[:, ::-1]  # (x,y) pairs

                num_clusters = np.ceil((np.prod(label_patch.size)/(args.tile_w*args.tile_h))+1).astype(np.int)
                kmeans = KMeans(n_clusters=num_clusters, random_state=0)
                cnt_pts = kmeans.fit(coords).cluster_centers_  # (x,y) centers
                cnt_pts = (us * cnt_pts).astype(np.int)

                '''
                import matplotlib.pyplot as plt
                plt.imshow(label_patch)
                plt.plot(cnt_pts[:, 0]//us, cnt_pts[:, 1]//us, 'o')
                plt.show()
                '''

                for _cx, _cy in cnt_pts:

                    _cx, _cy = np.asarray([_cx, _cy])

                    _cx = _cx + l
                    _cy = _cy + u

                    pwh = args.scan_resize * np.maximum(args.tile_w, args.tile_h)
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
                    tilepth_m = '{}/m_{}_{}.png'.format(args.train_image_pth, filename, patch_id)

                    ' save metadata '
                    metadata[filename][patch_id] = {
                        'wsi': tilepth_w,
                        'label': tilepth_g,
                    }

                    ' save images '
                    gt_patch = gt[up:down, left:right]
                    # gt_patch = (255 * np.eye(args.num_classes)[gt_patch][..., 1:]).astype(np.uint8)
                    gt_patch = Image.fromarray(gt_patch.astype(np.uint8))

                    if args.scan_resize != 1:
                        gt_patch = gt_patch.resize((args.tile_w, args.tile_h))

                    gt_patch.save('../' + tilepth_g)

                    wsi_patch = scan.read_region((int(left * scan.level_downsamples[args.scan_level]),
                                                  int(up * scan.level_downsamples[args.scan_level])),
                                                 args.scan_level,
                                                 (pwh, pwh)).convert('RGB')

                    if args.scan_resize != 1:
                        wsi_patch = wsi_patch.resize((args.tile_w, args.tile_h))
                    wsi_patch.save('../' + tilepth_w)

                    msk_patch = np.zeros(gt_patch.size[::-1], dtype=np.uint8)
                    msk_patch = Image.fromarray(msk_patch)

                    if args.scan_resize != 1:
                        msk_patch = msk_patch.resize((args.tile_w, args.tile_h))

                    msk_patch.save('../' + tilepth_m)

                    patch_id = patch_id + 1

                    if pwh not in pwhs:
                        pwhs[pwh] = 0

                    pwhs[pwh] += 1

        del gt
        del labels
        del scan


np.save(metadata_pth, metadata)

print(pwhs)
