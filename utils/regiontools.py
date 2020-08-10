'''
extract gt patches centered on patch
not random sliding windows
'''
import numpy as np
from sklearn.cluster import MiniBatchKMeans as KMeans
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix
from mahotas import bwperim
import cv2
import utils.dataset_hr as dhr


def map_points(arr, params):
    '''
    given an array of coordinates
    (+wsi end points x_max/y_max),
    maps points to zero level coordinates
    and if any of them are on the borders of
    the image, removes them
    '''

    arr = arr.astype(np.int)

    arr *= (4 ** params.scan_level)
    arr -= [params.tile_w // 2, params.tile_h // 2]

    'if centers are too close to the edge, discard the point'
    valid_indices = (arr[:, 0] > 0) * \
                    ((arr[:, 0] + params.tile_w) < params.iw) * \
                    (arr[:, 1] > 0) * \
                    ((arr[:, 1] + params.tile_h) < params.ih)

    arr = arr[valid_indices]

    return arr, arr.shape[0]


def remove_white_region(mask, arr, params, thresh=0.9):
    '''
    given an array of coordinates
    (+wsi end points x_max/y_max),
    finds patches that only/mostly
    have white, and removes it from
    the array
    thresh: % foreground required
    to keep the patch
    '''

    if arr is None or arr.shape[0] < 1:
        return None, 0

    tile_w = int(params.tile_w / (4 ** params.scan_level))
    tile_h = int(params.tile_h / (4 ** params.scan_level))

    valid_indices = np.zeros((arr.shape[0], ), dtype=np.bool)

    for ij, (x, y) in enumerate(arr):
        keep = np.count_nonzero(mask[y:y+tile_h, x:x+tile_w]) / (tile_h * tile_w) >= thresh
        valid_indices[ij] = keep

    arr = arr[valid_indices]

    return arr, arr.shape[0]


def get_key_points(image, us, min_clusters, max_clusters=9999999):

    if not isinstance(image, np.ndarray):
        image = np.asarray(image)

    image = Image.fromarray(image.astype(np.uint8))
    x, y = image.size
    image = image.resize((x//us, y//us))
    image = np.asarray(image)

    foreground_indices = np.nonzero(image)
    coords = np.transpose(foreground_indices)[:, ::-1]  # (x,y) pairs

    #num_clusters = np.power(coords.size/2, 1/3).astype(np.int)
    #num_clusters = np.maximum(min_clusters, num_clusters)
    #num_clusters = np.minimum(num_clusters, max_clusters)
    num_clusters = min_clusters

    if num_clusters <= 1 or coords.shape[0] <= 3 * num_clusters:
        return None, None, None, None

    km_out = KMeans(n_clusters=num_clusters, random_state=0).fit(coords)
    cnt_pts, cluster_assignments = km_out.cluster_centers_, km_out.labels_  # (x,y) centers
    cnt_pts = (us * cnt_pts).astype(np.int)

    out = np.zeros_like(image)
    out[foreground_indices] = cluster_assignments + 1

    out = Image.fromarray(out.astype(np.uint16))
    out = out.resize((x, y))
    out = np.asarray(out)

    foreground_indices = np.nonzero(out)

    return num_clusters, cnt_pts, out, foreground_indices


def get_key_points_for_patch(params):
    '''
    patches will only have image level
    label (no segmentation mask);
    hence no need for region
    level key points. instead, generate
    a roughly uniform
    point set.
    '''

    y_max = params.dimensions[1] // 4**params.scan_level
    x_max = params.dimensions[0] // 4**params.scan_level

    mask = np.zeros((y_max, x_max), dtype=np.uint8)

    y_min = 32
    x_min = 32

    mask[y_min:y_max-y_min, x_min:x_max-x_min] = 1

    perim = bwperim(mask)
    perim_coords = np.transpose(np.where(perim))[:, ::-1]  # (x,y) pairs
    skip = np.maximum(2, perim_coords.shape[0] // params.num_perim_points)
    perim_coords = perim_coords[::skip, :]

    kernel = np.ones((10, 10), np.uint8)

    _, center_pts, _, _ = get_key_points(cv2.erode(mask, kernel, iterations=1), 1, params.num_center_points, params.num_center_points)

    center_pts -= [params.tile_w // 2, params.tile_h // 2]
    perim_coords -= [params.tile_w // 2, params.tile_h // 2]

    return {
        'cnt_xy': center_pts,
        'perim_xy': perim_coords,
        'scan_level': params.scan_level
    }


def validate_dataset(model, dataset, epoch):
    '''
    given a validation dataset,
    predict class based on model
    '''

    model.eval()

    preds_c, preds_s, gts = [], [], []

    with torch.no_grad():

        for batch_it, (images, label) in enumerate(dataset):
            images = images.cuda()

            # pass images through the network (cls)
            pred_singles, pred_cls = model(images)

            pred_cls = torch.argmax(pred_cls, 1)
            preds_c.extend(pred_cls.cpu().numpy())

            '''
            ps_groups = torch.split(pred_singles, dhr.HR_NUM_PERIM_SAMPLES+dhr.HR_NUM_CNT_SAMPLES)
            ps_groups = [item.softmax(1).mean(0).argmax() for item in ps_groups]
            pred_singles = torch.tensor(ps_groups).cuda()
            preds_s.extend(pred_singles.cpu().numpy())
            '''

            gts.extend(label)

        preds_c = np.asarray(preds_c)
        # preds_s = np.asarray(preds_s)
        gts = np.asarray(gts)

        score_cls = (np.mean(preds_c == gts))
        cfs = confusion_matrix(gts, preds_c)
        cls_acc = np.diag(cfs/cfs.sum(1))  # classwise accuracy
        cls_acc = ['{:.2f}'.format(el) for el in cls_acc]

        '''
        score_singles = (np.mean(preds_s == gts))
        cfs = confusion_matrix(gts, preds_s)
        singles_acc = np.diag(cfs/cfs.sum(1))  # classwise accuracy
        singles_acc = ['{:.2f}'.format(el) for el in singles_acc]
        '''

        print('\n Epoch {}, '
              'Validation acc. {:.2f},'
              'Classwise acc. {}'
              #'\n'
              #'(s) Validation acc. {:.2f},'
              #'(s) Classwise acc. {}'
              ' \n'.format(
            epoch,
            score_cls,
            cls_acc,
            #score_singles,
            #singles_acc,
        ))

    model.train()
