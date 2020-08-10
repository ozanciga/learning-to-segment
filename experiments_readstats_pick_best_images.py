import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import glob
import os
from PIL import Image, ImageFont, ImageDraw
import re
from myargs import args
from argparse import Namespace
import cv2
from sklearn.metrics import f1_score

def metric(p, g):
    p = p.flatten()
    g = g.flatten()
    return np.mean(p[g > 0] == g[g > 0])


fontsize = 130
upleft = (35, 25)
downright = (300, 150)

bgcolor = 'gray'
textcolor = 'white'

criteria = [
    #'s_acc',
    's_acc_fg',
    #'s_iou',
    #'s_bach',
    #'s_f1mic',
    #'s_f1mac',
]

df = pd.read_csv('exp_results.csv', delimiter=',')

plot_pts = []

percent_seg = [0., 0.01, 0.025, 0.05, 0.075, 0.10]
exp_iters = [1, 2, 3, 4, 5]
seg_and_clss = [0, 1, 2]

o = {k: {0: np.array(7 * [0.0]), 1: np.array(7 * [0.0]), 2: np.array(7 * [0.0])} for k in percent_seg}
p = {k: {0: 7 * [''], 1: 7 * [''], 2: 7 * ['']} for k in percent_seg}
pths = {k: {0: {it: '' for it in exp_iters}, 1: {it: '' for it in exp_iters}, 2: {it: '' for it in exp_iters}} for k in percent_seg}

'get config pths (to extract images)'
exp_config = []
for exp_it, exp_pth in tqdm(enumerate(glob.glob('{}/*/'.format(args.experiments_pth)))):
    config = open('{}/config.txt'.format(exp_pth), 'r')
    cfg = config.read()
    config.close()

    cfg = eval(cfg.replace('array', ''))
    if cfg.use_seg_ratio in pths and cfg.seg_and_cls in pths[cfg.use_seg_ratio] and cfg.exp_iter in pths[cfg.use_seg_ratio][cfg.seg_and_cls]:
        pths[cfg.use_seg_ratio][cfg.seg_and_cls][cfg.exp_iter] = exp_pth

for use_seg_ratio in tqdm(percent_seg):
    for seg_and_cls in seg_and_clss:
        max_score_exp_iter_all = {k: len(exp_iters)*[0] for k in criteria}
        for exp_iter in exp_iters:
            if seg_and_cls == 0 and exp_iter > 1:continue
            for epoch in range(1, 1 + 1):
                stats = df[(df.exp_iter == exp_iter)
                           & (df.seg_and_cls == seg_and_cls)
                           & (df.use_seg_ratio == use_seg_ratio)
                           & (df.epoch == epoch)
                           & (df.case_id != 'A08')]

                np_stats = np.array(stats)

                outs = 0
                for k in criteria:
                    outs += np.array(stats[k]) / len(criteria)

                if len(outs) < 1 or len(outs) > 7:
                    continue

                conds = outs > o[use_seg_ratio][seg_and_cls]

                if outs.mean() >= o[use_seg_ratio][seg_and_cls].mean():
                    o[use_seg_ratio][seg_and_cls] = outs

                    for cj, c in enumerate(conds):

                        img_pth = glob.glob('{}/{}/{}'.format(
                            pths[use_seg_ratio][seg_and_cls][exp_iter],
                            epoch,
                            np_stats[cj][5]
                        ))[0]

                        p[use_seg_ratio][seg_and_cls][cj] = img_pth


for img_id in [os.path.basename(f).split('.svs')[0] for f in glob.glob('data/val/*.svs')]:

    args.gt_pths = 'data/val/gt_thumbnails'
    gt = np.array(Image.open('{}/{}.png'.format(args.gt_pths, img_id)))

    gt_cls = np.argmax(
        np.concatenate((np.zeros_like(gt[..., 0])[..., np.newaxis], gt), axis=2),
        axis=2
    ).flatten()


    g = np.array(Image.open('{}/{}.png'.format(args.gt_pths, img_id)).convert('L')) > 0
    g = cv2.morphologyEx(g.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((100, 100), np.uint8))[..., np.newaxis]

    gg = np.array(Image.fromarray(np.uint8(gt > 0) * 255).convert('L'))
    r, c = np.where(gg > 0)
    rmin, rmax, cmin, cmax = r.min(), r.max(), c.min(), c.max()

    image_0 = None
    image_1 = None
    image_2 = None

    for perc in percent_seg:
        for seg_and_cls in seg_and_clss:
            for it, val in enumerate(o[perc][seg_and_cls]):
                val = np.round(100 * val) / 100
                my_pth = p[perc][seg_and_cls][it]
                if img_id in my_pth:
                    if seg_and_cls == 0:

                        if image_0 is None:

                            img = Image.open(my_pth)
                            img = g * np.array(img)

                            pred_cls = np.argmax(
                                np.concatenate((np.zeros_like(gt[..., 0])[..., np.newaxis], img), axis=2),
                                axis=2
                            )
                            s_f1mic = metric(pred_cls, gt_cls)

                            img = img[rmin:rmax, cmin:cmax]

                            img = Image.fromarray(img)

                            d = ImageDraw.Draw(img)
                            fpth = '/usr/share/fonts/truetype/abyssinica/AbyssinicaSIL-R.ttf'
                            fnt = ImageFont.truetype(fpth, fontsize)
                            d.rectangle((upleft, downright), fill=bgcolor)
                            d.text(upleft, '{:.2f}'.format(s_f1mic), font=fnt, fill=textcolor)

                            image_0 = np.array(img)

                        else:

                            img = Image.open(my_pth)
                            img = g * np.array(img)

                            pred_cls = np.argmax(
                                np.concatenate((np.zeros_like(gt[..., 0])[..., np.newaxis], img), axis=2),
                                axis=2
                            )
                            s_f1mic = metric(pred_cls, gt_cls)

                            img = img[rmin:rmax, cmin:cmax]

                            img = Image.fromarray(img)

                            d = ImageDraw.Draw(img)
                            fpth = '/usr/share/fonts/truetype/abyssinica/AbyssinicaSIL-R.ttf'
                            fnt = ImageFont.truetype(fpth, fontsize)
                            d.rectangle((upleft, downright), fill=bgcolor)
                            d.text(upleft, '{:.2f}'.format(s_f1mic), font=fnt, fill=textcolor)

                            img = np.array(img)

                            image_0 = np.concatenate(
                                (image_0, img),
                                axis=1
                            )

                    elif seg_and_cls == 1:

                            if image_1 is None:

                                img = Image.open(my_pth)
                                img = g * np.array(img)

                                pred_cls = np.argmax(
                                    np.concatenate((np.zeros_like(gt[..., 0])[..., np.newaxis], img), axis=2),
                                    axis=2
                                )
                                s_f1mic = metric(pred_cls, gt_cls)

                                img = img[rmin:rmax, cmin:cmax]

                                img = Image.fromarray(img)

                                d = ImageDraw.Draw(img)
                                fpth = '/usr/share/fonts/truetype/abyssinica/AbyssinicaSIL-R.ttf'
                                fnt = ImageFont.truetype(fpth, fontsize)
                                d.rectangle((upleft, downright), fill=bgcolor)
                                d.text(upleft, '{:.2f}'.format(s_f1mic), font=fnt, fill=textcolor)

                                image_1 = np.array(img)

                            else:

                                img = Image.open(my_pth)
                                img = g * np.array(img)

                                pred_cls = np.argmax(
                                    np.concatenate((np.zeros_like(gt[..., 0])[..., np.newaxis], img), axis=2),
                                    axis=2
                                )

                                s_f1mic = metric(pred_cls, gt_cls)

                                img = img[rmin:rmax, cmin:cmax]

                                img = Image.fromarray(img)

                                d = ImageDraw.Draw(img)
                                fpth = '/usr/share/fonts/truetype/abyssinica/AbyssinicaSIL-R.ttf'
                                fnt = ImageFont.truetype(fpth, fontsize)
                                d.rectangle((upleft, downright), fill=bgcolor)
                                d.text(upleft, '{:.2f}'.format(s_f1mic), font=fnt, fill=textcolor)

                                img = np.array(img)

                                image_1 = np.concatenate(
                                    (image_1, img),
                                    axis=1
                                )

                    elif seg_and_cls == 2:

                        if image_2 is None:

                            img = Image.open(my_pth)
                            img = g * np.array(img)

                            pred_cls = np.argmax(
                                np.concatenate((np.zeros_like(gt[..., 0])[..., np.newaxis], img), axis=2),
                                axis=2
                            )
                            s_f1mic = metric(pred_cls, gt_cls)

                            img = img[rmin:rmax, cmin:cmax]

                            img = Image.fromarray(img)

                            d = ImageDraw.Draw(img)
                            fpth = '/usr/share/fonts/truetype/abyssinica/AbyssinicaSIL-R.ttf'
                            fnt = ImageFont.truetype(fpth, fontsize)
                            d.rectangle((upleft, downright), fill=bgcolor)
                            d.text(upleft, '{:.2f}'.format(s_f1mic), font=fnt, fill=textcolor)

                            image_2 = np.array(img)

                        else:

                            img = Image.open(my_pth)
                            img = g * np.array(img)

                            pred_cls = np.argmax(
                                np.concatenate((np.zeros_like(gt[..., 0])[..., np.newaxis], img), axis=2),
                                axis=2
                            )
                            s_f1mic = metric(pred_cls, gt_cls)

                            img = img[rmin:rmax, cmin:cmax]

                            img = Image.fromarray(img)

                            d = ImageDraw.Draw(img)
                            fpth = '/usr/share/fonts/truetype/abyssinica/AbyssinicaSIL-R.ttf'
                            fnt = ImageFont.truetype(fpth, fontsize)
                            d.rectangle((upleft, downright), fill=bgcolor)
                            d.text(upleft, '{:.2f}'.format(s_f1mic), font=fnt, fill=textcolor)

                            img = np.array(img)

                            image_2 = np.concatenate(
                                (image_2, img),
                                axis=1
                            )

    image_0 = np.concatenate((gt[rmin:rmax, cmin:cmax], image_0), axis=1)
    image_1 = np.concatenate((np.zeros_like(gt[rmin:rmax, cmin:cmax]), image_1), axis=1)
    image_2 = np.concatenate((np.zeros_like(gt[rmin:rmax, cmin:cmax]), image_2), axis=1)

    image = np.concatenate((image_0, image_1, image_2), axis=0)
    image[(image[..., 0] == 0) * (image[..., 1] == 0) * (image[..., 2] == 0)] = 255

    image = Image.fromarray(image)

    x, y = image.size

    image = image.resize((x // 4, y // 4))

    image.save('{}_cat.png'.format(img_id))

