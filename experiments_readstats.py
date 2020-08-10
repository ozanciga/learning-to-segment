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
with open('exp_results.csv', 'w') as csvfile:
    fieldnames = [
        'epoch',
        'exp_it', 'exp_iter', 'seg_and_cls', 'use_seg_ratio',
        'case_id',
        's_acc', 's_acc_fg', 's_iou', 's_bach', 's_f1mic', 's_f1mac',
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
    g = np.concatenate((np.zeros((*gt.size[::-1], 1)), np.array(gt)), axis=-1)
    g = np.argmax(g, axis=-1)

    gt_image[filename] = g
    gt_image_flatten[filename] = g.flatten()


def image_stats(g_filename, pred_image):
    #p = np.concatenate((np.zeros((*pred_image.size[::-1], 1)), np.array(pred_image)), axis=-1)
    #p = np.argmax(p, axis=-1)
    p = np.zeros(pred_image.size[::-1], dtype=np.uint8)
    pred_image = np.array(pred_image)
    for cj in range(3):
        p[pred_image[..., cj] > 0] = cj+1

    g = gt_image[g_filename]
    gf = gt_image_flatten[g_filename]

    s_bach = 1 - np.sum(np.abs(p - g)) / np.sum(
        np.maximum(np.abs(g - 0), np.abs(g - 3.0)) * (1 - (1 - (p > 0)) * (1 - g > 0)))

    pf = p.flatten()
    s_f1mic = metrics.f1_score(pf, gf, average='micro')
    s_f1mac = metrics.f1_score(pf, gf, average='macro')

    iou = 0
    for cj in range(1, args.num_classes):
        pj = (p == cj).astype(np.uint8)
        gj = (g == cj).astype(np.uint8)
        inters = (pj * gj).sum()
        union = (pj | gj).sum()
        iou += inters / union if union > 0 else 1.0

    s_acc = np.mean(p == g)
    s_acc_fg = np.mean(p[g > 0] == g[g > 0])

    s_iou = (iou / (args.num_classes - 1))

    return s_acc, s_acc_fg, s_iou, s_bach, s_f1mic, s_f1mac


def folder_stats(exp_config, exp_pth):
    epoch_pths = glob.glob('{}/*/'.format(exp_pth))

    csvfile = open('exp_results.csv', 'a')
    fieldnames = [
        'epoch',
        'exp_it', 'exp_iter', 'seg_and_cls', 'use_seg_ratio',
        'case_id',
        's_acc', 's_acc_fg', 's_iou', 's_bach', 's_f1mic', 's_f1mac',
    ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    for ep in (sorted(epoch_pths)):
        pred_pths = sorted(glob.glob('{}/*.png'.format(ep)))
        for pred_pth in pred_pths:
            filename = os.path.basename(pred_pth)
            #filename = re.findall('(.+).svs', filename)[0]
            #pred_image = Image.open(pred_pth)
            #stats = image_stats(filename, pred_image)
            stats = [float(i) for i in filename.split('_overlay_')[-1].replace('.png', '').split('_')]


            writer.writerow({
                'epoch': int(re.findall('/([0-9]+)/', ep)[0]),
                'exp_it': exp_config[0],
                'exp_iter': exp_config[1],
                'seg_and_cls': exp_config[2],
                'use_seg_ratio': exp_config[3],
                'case_id': filename,
                's_acc': stats[0],
                's_acc_fg': stats[1],
                's_iou': stats[2],
                's_bach': stats[3],
                's_f1mic': stats[4],
                's_f1mac': stats[5],
            })

    csvfile.close()


exp_config = []
for exp_it, exp_pth in tqdm(enumerate(glob.glob('{}/*/'.format(args.experiments_pth)))):
    config = open('{}/config.txt'.format(exp_pth), 'r')
    cfg = config.read()
    config.close()

    cfg = eval(cfg.replace('array', ''))
    exp_config = (exp_it, cfg.exp_iter, cfg.seg_and_cls, cfg.use_seg_ratio)

    folder_stats(exp_config, exp_pth)


