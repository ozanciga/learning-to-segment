import torch
import torch.nn.functional as F
import os
from myargs import args
from utils import preprocessing
import numpy as np
from PIL import Image
import cv2
import gc
import openslide
from mahotas import bwperim
from skimage.morphology.convex_hull import convex_hull_image as chull
import glob
import torchvision
from sklearn.metrics import f1_score
from sklearn import metrics
import matplotlib.pyplot as plt
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None

def predict_wsis(model, dataset, ep):
    '''
    given directory svs_path,
    current model goes through each
    wsi (svs) and generates a
    tumor bed.
    '''

    # os.makedirs('{}/{}'.format(args.val_save_pth, ep), exist_ok=True)
    os.makedirs('{}/{}'.format(args.experiment_save_pth, ep))

    model.eval()

    acc, acc_fg, scores, scores_f1mic, scores_bach, scores_f1mac = [], [], [], [], [], []

    with torch.no_grad():
        ' go through each svs and make a pred. mask'
        with tqdm(enumerate(dataset.wsis), disable=True) as t:
            for wj, key in t:

                t.set_description(
                    'Generating heatmaps of tumor beds from wsis.. {:d}/{:d}'.format(1 + wj, len(dataset.wsis)))

                scan = dataset.wsis[key]['scan']

                'create prediction template'
                wsi_yx_dims = scan.level_dimensions[2][::-1]
                pred = np.zeros((args.num_classes, *wsi_yx_dims), dtype=np.float)
                'downsample multiplier'
                m = scan.level_downsamples[args.scan_level] / scan.level_downsamples[2]

                dx, dy = int(m * dataset.params.pw), int(m * dataset.params.ph)

                'slide over wsi'

                for batch_x, batch_y, batch_image in dataset.wsis[key]['iterator']:

                    batch_image = batch_image.cuda()

                    feature_maps = model(batch_image)
                    _, pred_src = model.c1(feature_maps)

                    if args.scan_resize != 1:
                        pred_src = F.interpolate(
                            pred_src,
                            (args.tile_h * args.scan_resize, args.tile_w * args.scan_resize)
                        )
                    if args.scan_level != 2:
                        pred_src = F.interpolate(
                            pred_src,
                            (int(m * args.tile_h), int(m * args.tile_h))
                        )

                    pred_src = pred_src.cpu().numpy()

                    while pred.ndim >= pred_src.ndim:
                        pred_src = np.expand_dims(pred_src, -1)

                    for bj in range(batch_image.size(0)):
                        tile_x, tile_y = int(m * batch_x[bj]), int(m * batch_y[bj])
                        pred[:, tile_y:tile_y + dy, tile_x:tile_x + dx] += pred_src[bj]

                pred_classes, pred_heatmap = preprocessing.threshold_probs(pred)

                'save overlay mask'
                #pred_image = scan.read_region((0, 0), 2, scan.level_dimensions[2]).convert('RGB')
                #pred_image = np.asarray(pred_image).astype(np.uint8)

                msk_pth = '{}/{}.png'.format(args.wsi_mask_pth, key)
                if not os.path.exists(msk_pth):
                    thmb = scan.read_region((0, 0), 2, scan.level_dimensions[2]).convert('RGB')
                    mask = preprocessing.find_nuclei(thmb, fill_mask=True)
                    Image.fromarray(mask.astype(np.uint8)).save(msk_pth)
                else:
                    mask = Image.open(msk_pth).convert('L')
                    mask = np.asarray(mask)

                pred_classes = pred_classes * mask
                pred_classes = np.eye(args.num_classes)[pred_classes][..., 1:]

                pred_image = 255 * pred_classes
                pred_image = Image.fromarray(np.uint8(pred_image))

                gt_image = Image.open('data/val/gt_thumbnails/{}.png'.format(key.replace('.svs', '')))
                x, y = gt_image.size

                pred_image = pred_image.resize((x, y))

                p = np.concatenate((np.zeros((*pred_image.size[::-1], 1)), np.array(pred_image)), axis=-1)
                p = np.argmax(p, axis=-1)
                g = np.concatenate((np.zeros((*gt_image.size[::-1], 1)), np.array(gt_image)), axis=-1)
                g = np.argmax(g, axis=-1)

                # scores_cfs.append(metrics.confusion_matrix(g.flatten(), p.flatten()))

                s = 1 - np.sum(np.abs(p - g)) / np.sum(
                    np.maximum(np.abs(g - 0), np.abs(g - 3.0)) * (1 - (1 - (p > 0)) * (1 - g > 0)))

                scores_bach.append(s)

                f1mic = metrics.f1_score(p.flatten(), g.flatten(), average='micro')
                f1mac = metrics.f1_score(p.flatten(), g.flatten(), average='macro')

                scores_f1mic.append(f1mic)
                scores_f1mac.append(f1mac)

                iou = 0
                for cj in range(1, args.num_classes):
                    pj = (p == cj).astype(np.uint8)
                    gj = (g == cj).astype(np.uint8)
                    inters = (pj * gj).sum()
                    union = (pj | gj).sum()
                    iou += inters/union if union > 0 else 1.0

                acc.append(np.mean(p == g))
                acc_fg.append(np.mean(p[g>0] == g[g>0]))

                iou = iou / (args.num_classes - 1)

                scores.append(iou)

                pred_image.save('{}/{}/{}_{}_overlay_{:.2f}_{:.2f}_{:.2f}_{:.2f}_{:.2f}_{:.2f}.png'.format(args.experiment_save_pth, ep, key, args.tile_stride_w, acc[0], acc_fg[0], iou, s, f1mac, f1mic))

                del pred
                del pred_image

    # print('Epoch {}'.format(ep))
    '''print('Ep {}, '
          'Acc {:.2f}, '
          'Fg {:.2f}, '
          'IOU {:.2f}, '
          'Bach {:.2f}, '
          'F1 mic {:.2f}, '
          'F1 mac {:.2f}, '.format(
            ep,
            np.mean(acc),
            np.mean(acc_fg),
            np.mean(scores),
            np.mean(scores_bach),
            np.mean(scores_f1mic),
            np.mean(scores_f1mac),
        ))'''

    model.train()


def predict_cls(model, dataset, ep):

    os.makedirs('{}/{}'.format(args.experiment_save_pth, ep))

    model.eval()

    with torch.no_grad():

        preds, gts = [], []

        for batch_it, (image, label) in enumerate(dataset):

            image = image.cuda()

            f = model(image)
            #feature_maps_l = f
            feature_maps_l, _ = model.c1(f)

            pred_cls = model.fc(feature_maps_l)

            pred_cls = torch.argmax(pred_cls, 1)

            preds.extend(pred_cls.cpu().numpy())
            gts.extend(label.numpy())

    preds, gts = np.asarray(preds), np.asarray(gts)

    np.save('{}/{}/preds.npy'.format(args.experiment_save_pth, ep), preds)
    np.save('{}/{}/gts.npy'.format(args.experiment_save_pth, ep), gts)

    print('Ep {:d}, Acc {:.2f}'.format(
        ep,
        np.mean(preds == gts)
    ))

    model.train()
