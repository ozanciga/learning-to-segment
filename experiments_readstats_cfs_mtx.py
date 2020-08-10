import glob
import os
from PIL import Image
from sklearn import metrics
import matplotlib.pyplot as plt
from tqdm import tqdm
from myargs import args
from argparse import Namespace
import numpy as np
import re
import csv

'save results in a csv'
with open('exp_results_cfs.csv', 'w') as csvfile:
    fieldnames = [
        'epoch',
        'exp_it', 'exp_iter', 'seg_and_cls', 'use_seg_ratio',
        'case_id',
        's_cfs',
    ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    csvfile.close()

'read in gt'
gt_paths = 'data/val/gt_thumbnails/'
gt_image = {}
gt_image_flatten = {}
for gt_pth in glob.glob('{}/*.png'.format(gt_paths)):
    filename = os.path.basename(gt_pth)
    filename = re.findall('(.+).png', filename)[0]

    gt = Image.open(gt_pth)
    #g = np.concatenate((np.zeros((*gt.size[::-1], 1)), np.array(gt)), axis=-1)
    #g = np.argmax(g, axis=-1)
    g = np.array(gt) > 0

    gt_image[filename] = g
    gt_image_flatten[filename] = g.flatten()


def image_stats(g_filename, pred_image):
    #p = np.zeros(pred_image.size[::-1], dtype=np.uint8)
    pred_image = np.array(pred_image) > 0
    #for cj in range(3):
    #    p[pred_image[..., cj] > 0] = cj+1

    g = gt_image[g_filename]
    '''gf = gt_image_flatten[g_filename]
    pf = p.flatten()

    s_cfs = metrics.confusion_matrix(gf, pf, labels=[0, 1, 2, 3])
    n_cfs = s_cfs.astype('float') / s_cfs.sum(axis=1)[:, np.newaxis]
    n_cfs[np.isnan(n_cfs)] = 0
    n_cfs = np.diag(n_cfs)'''

    n_cfs = [metrics.f1_score((g[..., cj]).flatten(), (pred_image[..., cj]).flatten()) for cj in range(3)]
    n_cfs = [0., *n_cfs]

    return (n_cfs)


def folder_stats(exp_config, exp_pth):
    epoch_pths = glob.glob('{}/*/'.format(exp_pth))

    csvfile = open('exp_results_cfs.csv', 'a')
    fieldnames = [
        'epoch',
        'exp_it', 'exp_iter', 'seg_and_cls', 'use_seg_ratio',
        'case_id',
        's_cfs',
    ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    for ep in (sorted(epoch_pths)):
        pred_pths = sorted(glob.glob('{}/*.png'.format(ep)))
        for pred_pth in pred_pths:
            filename = os.path.basename(pred_pth)
            filename = re.findall('(.+).svs', filename)[0]
            pred_image = Image.open(pred_pth)

            stats = image_stats(filename, pred_image)

            writer.writerow({
                'epoch': int(re.findall('/([0-9]+)/', ep)[0]),
                'exp_it': exp_config[0],
                'exp_iter': exp_config[1],
                'seg_and_cls': exp_config[2],
                'use_seg_ratio': exp_config[3],
                'case_id': filename,
                's_cfs': stats,
            })

    csvfile.close()


exp_config = []
for exp_it, exp_pth in enumerate(tqdm(glob.glob('{}/*/'.format(args.experiments_pth)))):
    config = open('{}/config.txt'.format(exp_pth), 'r')
    cfg = config.read()
    config.close()

    cfg = eval(cfg.replace('array', ''))
    exp_config = (exp_it, cfg.exp_iter, cfg.seg_and_cls, cfg.use_seg_ratio)

    folder_stats(exp_config, exp_pth)


