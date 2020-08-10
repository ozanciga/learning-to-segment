from PIL import Image
import numpy as np
import openslide
from skimage.morphology.convex_hull import convex_hull_image
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
import utils.dataset_hr as dhr
from concave_hull.ConcaveHull import concaveHull
from contour_ordering import evenly_spaced_points_on_a_contour as esp

train_hr_pths = [
    args.train_hr_image_pth,
    'data/val_hr',
    ]
raw_train_pths = [
    args.raw_train_pth,
    'data/val',
    ]

us_kmeans = 8  # undersample the gt for faster processing
scan_level = 2  # of which we extract coordinates
min_center_points = dhr.HR_NUM_CNT_SAMPLES

for ij in range(2):

    ufs.make_folder('../' + train_hr_pths[ij], True)
    wsipaths = sorted(glob.glob('../{}/*.svs'.format(raw_train_pths[ij])))

    ' check if metadata gt.npy already exists to append to it '
    metadata_pth = '../{}/gt.npy'.format(train_hr_pths[ij])
    metadata = ufs.fetch_metadata(metadata_pth)

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
        _, labels, _, _ = cv2.connectedComponentsWithStats((gt > 0).astype(np.uint8))

        ' save metadata (whole-slide)'
        metadata[filename] = {}

        for tile_id in range(1, labels.max()+1):

            label_patch = labels == tile_id

            area = np.count_nonzero(label_patch)
            num_clusters = dhr.HR_NUM_CNT_SAMPLES  # + int(area / (0.01 * gt.size))

            current_label = stats.mode(gt[label_patch])[0][0]

            'get width & height'
            indices = np.nonzero(label_patch)
            h = 1 + indices[0].max() - indices[0].min()
            w = 1 + indices[1].max() - indices[1].min()

            if 1 or (w * h) / gt.size <= 0.005:

                n, center_pts, out_image, _ = regiontools.get_key_points(label_patch, us_kmeans, dhr.HR_NUM_CNT_SAMPLES, num_clusters)

                if n is not None:

                    perim_coords = np.zeros([0, 2])
                    if dhr.HR_NUM_PERIM_SAMPLES > 0:
                        '''
                        this only contains region perimeter points, particularly useful
                        if the border information is a good signal for deciding class
                        '''

                        '''
                        perim = bwperim(convex_hull_image(label_patch))  # perim = bwperim(label_patch)
                        perim_coords = np.transpose(np.where(perim))[:, ::-1]  # (x,y) pairs
                        skip = np.maximum(2, perim_coords.shape[0] // dhr.HR_NUM_PERIM_SAMPLES)
                        perim_coords = perim_coords[::skip, :]
                        '''
                        label_patch = Image.fromarray(label_patch.astype(np.uint8))
                        x, y = label_patch.size
                        label_patch = label_patch.resize((x // us_kmeans, y // us_kmeans))
                        perim = bwperim(np.asarray(label_patch))
                        coords_ = np.transpose(np.where(perim))[:, ::-1]  # (x,y) pairs
                        cvh = concaveHull(coords_, 3)
                        perim_coords = esp(cvh, dhr.HR_NUM_PERIM_SAMPLES) * us_kmeans

                    '''
                    cluster centers/points within the region
                    that are equally spaced inside the region
                    '''
                    if tile_id not in metadata[filename]:
                        metadata[filename][tile_id] = {}

                    metadata[filename][tile_id][0] = {
                        'cnt_xy': center_pts,
                        'perim_xy': perim_coords,
                        'label': current_label,
                        'wsipath': wsipath,
                        'scan_level': scan_level,
                    }

            else:

                n, cnt_pts, out_image, foreground_indices = regiontools.get_key_points(label_patch, us_kmeans,
                                                                                       num_clusters, num_clusters)

                if n is not None:

                    ' go further in if region is large '
                    for r_id in range(1, n+1):

                        sub_patch = (out_image == r_id)

                        sub_n, sub_center_pts, _, sub_foreground_indices = regiontools.get_key_points(sub_patch, us_kmeans, min_center_points)

                        if sub_n is None or (
                                tile_id == 0 and
                                np.count_nonzero(wsi_mask[sub_foreground_indices]) / (sub_foreground_indices[0].shape[0]) < 0.9):
                            continue

                        sub_perim_coords = np.zeros([0, 2])
                        if dhr.HR_NUM_PERIM_SAMPLES > 0:

                            '''
                            perim = bwperim(convex_hull_image(sub_patch))
                            sub_perim_coords = np.transpose(np.where(perim))[:, ::-1]  # (x,y) pairs
                            skip = np.maximum(2, sub_perim_coords.shape[0] // dhr.HR_NUM_PERIM_SAMPLES)
                            sub_perim_coords = sub_perim_coords[::skip, :]
                            '''
                            sub_patch = Image.fromarray(sub_patch.astype(np.uint8))
                            x, y = sub_patch.size
                            sub_patch = sub_patch.resize((x // us_kmeans, y // us_kmeans))
                            perim = bwperim(np.asarray(sub_patch))
                            coords_ = np.transpose(np.where(perim))[:, ::-1]  # (x,y) pairs
                            cvh = concaveHull(coords_, 3)
                            sub_perim_coords = esp(cvh, dhr.HR_NUM_PERIM_SAMPLES) * us_kmeans

                        if tile_id not in metadata[filename]:
                            metadata[filename][tile_id] = {}

                        metadata[filename][tile_id][r_id] = {
                            'cnt_xy': sub_center_pts,
                            'perim_xy': sub_perim_coords,
                            'label': current_label,
                            'wsipath': wsipath,
                            'scan_level': scan_level,
                        }

    np.save(metadata_pth, metadata)
