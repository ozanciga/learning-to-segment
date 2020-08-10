from PIL import Image
import numpy as np
import openslide
from utils import preprocessing
import cv2
from myargs import args
from mahotas import bwperim
from utils import regiontools
import os
from utils.read_xml import getGT
import glob
import utils.filesystem as ufs
from scipy import stats
from tqdm import tqdm
from skimage.segmentation import slic
from skimage.util import img_as_float
import utils.dataset_hr as dhr


train_hr_pths = [
    'data/val_hr',
    args.train_hr_image_pth
    ]
raw_train_pths = [
    'data/val',
    args.raw_train_pth
    ]

for ij in range(2):

    ufs.make_folder('../' + train_hr_pths[ij], True)
    wsipaths = sorted(glob.glob('../{}/*.svs'.format(raw_train_pths[ij])))

    ' check if metadata gt.npy already exists to append to it '
    metadata_pth = '../{}/gt.npy'.format(train_hr_pths[ij])
    metadata = ufs.fetch_metadata(metadata_pth)

    us_kmeans = 4  # undersample the gt for faster processing
    scan_level = 2  # of which we extract coordinates
    numSegments = 1000
    sigma = 5
    compactness = 20

    for wsipath in tqdm(wsipaths):

        filename = os.path.basename(wsipath)

        'open scan'
        scan = openslide.OpenSlide(wsipath)
        wsi = scan.read_region((0, 0), scan_level, scan.level_dimensions[scan_level]).convert('RGB')

        'get actual mask, i.e. the ground truth'
        xmlpath = '../{}/{}.xml'.format(raw_train_pths[ij], filename.split('.svs')[0])
        gt = getGT(xmlpath, scan, level=args.scan_level)

        'generate color thresholded wsi mask'
        x, y = wsi.size
        wsi_small = wsi.resize((x // 4, y // 4))
        wsi_mask = preprocessing.find_nuclei(wsi_small)
        wsi_mask = Image.fromarray(wsi_mask)
        wsi_mask = wsi_mask.resize((x, y))
        wsi_mask = np.asarray(wsi_mask)

        'examine each connected component/training instance'
        labels = slic(img_as_float(wsi_small),
                      n_segments=numSegments,
                      sigma=sigma,
                      compactness=compactness)

        labels = Image.fromarray(labels.astype(np.uint16))
        labels = labels.resize((x, y))
        labels = np.asarray(labels)

        ' save metadata (whole-slide)'
        metadata[filename] = {}
        metadata[filename][0] = {}

        for tile_id in range(1 + labels.max()):

            label_patch = labels == tile_id

            n, center_pts, out_image, foreground_indices = regiontools.get_key_points(label_patch, us_kmeans,
                                           dhr.HR_NUM_CNT_SAMPLES, dhr.HR_NUM_CNT_SAMPLES)

            current_label = stats.mode(gt[label_patch])[0][0]

            if current_label < 1 and np.count_nonzero(wsi_mask[foreground_indices]) / (foreground_indices[0].shape[0]) < 0.9:
                continue

            perim_coords = np.zeros([0, 2])
            if dhr.HR_NUM_PERIM_SAMPLES > 0:
                perim = bwperim(label_patch)
                perim_coords = np.transpose(np.where(perim))[:, ::-1]  # (x,y) pairs
                skip = np.maximum(2, perim_coords.shape[0] // dhr.HR_NUM_PERIM_SAMPLES)
                perim_coords = perim_coords[::skip, :]

            metadata[filename][0][tile_id] = {
                'cnt_xy': center_pts,
                'perim_xy': perim_coords,
                'wsipath': wsipath,
                'label': current_label,
                'scan_level': scan_level,
                'foreground_indices': foreground_indices,
                'tile_id': tile_id,
            }

    np.save(metadata_pth, metadata)
