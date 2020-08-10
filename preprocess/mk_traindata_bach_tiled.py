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
import openslide
import os
import numpy as np
from myargs import args
import glob
from utils.read_xml import getGT
import utils.filesystem as ufs
from tqdm import tqdm
from PIL import Image
import gc
from utils.preprocessing import nextpow2, find_nuclei, tile_image, isforeground, stain_normalize, DotDict

istrain = True
if 0:
    args.train_image_pth = args.val_image_pth
    args.raw_train_pth = '../' + args.val_image_pth
    istrain = False
    args.tile_stride_w = args.tile_w
    args.tile_stride_h = args.tile_h

def save_patch(filename, patch_id, metadata, gt_patch, scan, xpos, ypos):
    'patch paths'
    tilepth_w = '{}/w_{}_{}.png'.format(args.train_image_pth, filename, patch_id)
    tilepth_g = '{}/g_{}_{}.png'.format(args.train_image_pth, filename, patch_id)

    ' save images '
    label = np.long(np.median(gt_patch))

    gt_patch = Image.fromarray(gt_patch.astype(np.uint8))

    wsi_patch = scan.read_region((int(xpos * scan.level_downsamples[2]), int(ypos * scan.level_downsamples[2])),
                                 args.scan_level,
                                 (args.tile_w, args.tile_h)).convert('RGB')

    wsi_patch.save('../' + tilepth_w)

    ' save metadata '
    gt_patch.save('../' + tilepth_g)
    metadata[filename][patch_id] = {
        'wsi': tilepth_w,
        'label': label,
        'mask': tilepth_g,
    }

    patch_id = patch_id + 1

    return metadata, patch_id


ufs.make_folder('../' + args.train_image_pth, istrain)
wsipaths = glob.glob('{}/*.svs'.format(args.raw_train_pth))

' check if metadata gt.npy already exists to append to it '
metadata_pth = '../{}/gt.npy'.format(args.train_image_pth)
metadata = ufs.fetch_metadata(metadata_pth)

wsipaths = sorted(wsipaths)
patch_id = 0

for wj, wsipath in tqdm(enumerate(wsipaths)):

    'read scan and get metadata'
    scan = openslide.OpenSlide(wsipath)
    filename = os.path.basename(wsipath)
    metadata[filename] = {}

    params = {
        'iw': scan.level_dimensions[args.scan_level][0],
        'ih': scan.level_dimensions[args.scan_level][1],
        'sw': args.tile_stride_w,
        'sh': args.tile_stride_h,
        'ph': args.tile_h * args.scan_resize,  # patch height (y)
        'pw': args.tile_w * args.scan_resize,  # patch width (x)

    }
    params = DotDict(params)

    m_lvl = scan.level_downsamples[2] / scan.level_downsamples[args.scan_level]  # level multiplier

    'get actual mask, i.e. the ground truth'
    xmlpath = '{}/{}.xml'.format(args.raw_train_pth, filename.split('.svs')[0])

    gt = getGT(xmlpath, scan, level=2)
    #gt = (255 * np.eye(args.num_classes)[gt][..., 1:]).astype(np.uint8)

    msk_pth = '{}/{}.png'.format(args.wsi_mask_pth, filename)
    if not os.path.exists(msk_pth):
        thmb = scan.read_region((0, 0), 2, scan.level_dimensions[2]).convert('RGB')
        mask = find_nuclei(thmb, fill_mask=True)
        Image.fromarray(mask.astype(np.uint8)).save(msk_pth)
    else:
        mask = Image.open(msk_pth).convert('L')
        mask = np.asarray(mask)

    'downsample multiplier'
    m = scan.level_downsamples[args.scan_level] / scan.level_downsamples[2]
    dx, dy = int(params.pw * m), int(params.ph * m)

    for ypos in range(1, params.ih - 1 - params.ph, params.sh):
        for xpos in range(1, params.iw - 1 - params.pw, params.sw):
            yp, xp = int(ypos * m), int(xpos * m)
            if isforeground(mask[yp:yp + dy, xp:xp + dx]):
                metadata, patch_id = save_patch(filename, patch_id, metadata, gt[yp:yp + dy, xp:xp + dx], scan, xpos, ypos)

    xpos = params.iw - 1 - params.pw
    for ypos in range(1, params.ih - 1 - params.ph, params.sh):
        yp, xp = int(ypos * m), int(xpos * m)
        if isforeground(mask[yp:yp + dy, xp:xp + dx]):
            metadata, patch_id = save_patch(filename, patch_id, metadata, gt[yp:yp + dy, xp:xp + dx], scan, xpos, ypos)

    ypos = params.ih - 1 - params.ph
    for xpos in range(1, params.iw - 1 - params.pw, params.sw):
        yp, xp = int(ypos * m), int(xpos * m)
        if isforeground(mask[yp:yp + dy, xp:xp + dx]):
            metadata, patch_id = save_patch(filename, patch_id, metadata, gt[yp:yp + dy, xp:xp + dx], scan, xpos, ypos)

    del gt
    del scan

np.save(metadata_pth, metadata)
