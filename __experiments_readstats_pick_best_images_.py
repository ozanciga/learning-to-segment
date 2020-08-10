import glob
import os
from PIL import Image, ImageDraw, ImageFont
from sklearn import metrics
import matplotlib.pyplot as plt
from tqdm import tqdm
from myargs import args
from argparse import Namespace
import numpy as np
import cv2
import re
from skimage.morphology.convex_hull import convex_hull_image

fontsize = 130
upleft = (35, 25)
downright = (300, 150)

bgcolor = 'gray'
textcolor = 'white'

# acc, acc_fg, iou, s, f1mac, f1mic
criterion = 2

for img_id in ['A04', 'A05', 'A06', 'A07', 'A08', 'A09', 'A10']:

    percent_seg = [0.01, 0.025, 0.05]

    args.gt_pths = 'data/val/gt_thumbnails'
    gt = np.array(Image.open('{}/{}.png'.format(args.gt_pths, img_id)))
    g = np.array(Image.open('{}/{}.png'.format(args.gt_pths, img_id)).convert('L')) >= 0
    g = cv2.morphologyEx(g.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))[..., np.newaxis]

    gg = np.array(Image.fromarray(np.uint8(gt > 0) * 255).convert('L'))
    r, c = np.where(gg > 0)
    rmin, rmax, cmin, cmax = r.min(), r.max(), c.min(), c.max()

    def folder_stats(exp_pth):
        epoch_pths = glob.glob('{}/*/'.format(exp_pth))

        outs = np.zeros((1, 7))
        pths = 7 * ['']

        for ej, ep in enumerate(sorted(epoch_pths)):
            out = []
            pred_pths = sorted(glob.glob('{}/*.png'.format(ep)))
            for pred_pth in pred_pths:
                filename = os.path.basename(pred_pth)
                stats = filename.replace('.png', '').split('_')
                stats = [np.float(it) for it in stats[3:]]
                out.append(stats)

            # acc, acc_fg, iou, s, f1mac, f1mic
            out = np.array(out)[:, criterion].reshape(1, -1)
            conds = out >= outs
            outs[conds] = out[conds]
            for cj, c in enumerate(conds[0]):
                if c:
                    pths[cj] = pred_pths[cj]

        return pths, outs


    o = {k: {0: np.array(7*[0.0]), 1: np.array(7*[0.0])} for k in percent_seg}
    p = {k: {0: 7*[''], 1: 7*['']} for k in percent_seg}

    for exp_it, exp_pth in tqdm(enumerate(glob.glob('{}/*/'.format(args.experiments_pth))), disable=True):

        config = open('{}/config.txt'.format(exp_pth), 'r')
        cfg = config.read()
        config.close()

        cfg = eval(cfg.replace('array', ''))

        if cfg.exp_iter != 1 or cfg.use_seg_ratio not in percent_seg:
            continue

        pths, outs = folder_stats(exp_pth)

        outs = outs[0]

        conds = outs > o[cfg.use_seg_ratio][cfg.seg_and_cls]

        for cj, c in enumerate(conds):
            if c:
                o[cfg.use_seg_ratio][cfg.seg_and_cls][cj] = outs[cj]
                p[cfg.use_seg_ratio][cfg.seg_and_cls][cj] = pths[cj]

    image_0 = None
    image_1 = None

    for perc in percent_seg:
        for seg_and_cls in [0, 1]:
            print('************{:.2f}, {:d}************'.format(perc, seg_and_cls))
            for it, val in enumerate(o[perc][seg_and_cls]):
                val = np.round(100 * val) / 100
                my_pth = p[perc][seg_and_cls][it]
                if img_id in my_pth:
                    #print(val, '-', my_pth)
                    if seg_and_cls == 0:

                        if image_0 is None:

                            img = Image.open(my_pth)
                            img = g * np.array(img)
                            img = img[rmin:rmax, cmin:cmax]

                            img = Image.fromarray(img)

                            d = ImageDraw.Draw(img)
                            fpth = '/usr/share/fonts/truetype/abyssinica/AbyssinicaSIL-R.ttf'
                            fnt = ImageFont.truetype(fpth, fontsize)
                            d.rectangle((upleft, downright), fill=bgcolor)
                            d.text(upleft, str(val), font=fnt, fill=textcolor)

                            image_0 = np.array(img)

                        else:

                            img = Image.open(my_pth)
                            img = g * np.array(img)
                            img = img[rmin:rmax, cmin:cmax]

                            img = Image.fromarray(img)

                            d = ImageDraw.Draw(img)
                            fpth = '/usr/share/fonts/truetype/abyssinica/AbyssinicaSIL-R.ttf'
                            fnt = ImageFont.truetype(fpth, fontsize)
                            d.rectangle((upleft, downright), fill=bgcolor)
                            d.text(upleft, str(val), font=fnt, fill=textcolor)

                            img = np.array(img)

                            image_0 = np.concatenate(
                                (image_0, img),
                                axis=1
                            )
                    else:

                        if image_1 is None:

                            img = Image.open(my_pth)
                            img = g * np.array(img)
                            img = img[rmin:rmax, cmin:cmax]

                            img = Image.fromarray(img)

                            d = ImageDraw.Draw(img)
                            fpth = '/usr/share/fonts/truetype/abyssinica/AbyssinicaSIL-R.ttf'
                            fnt = ImageFont.truetype(fpth, fontsize)
                            d.rectangle((upleft, downright), fill=bgcolor)
                            d.text(upleft, str(val), font=fnt, fill=textcolor)

                            image_1 = np.array(img)

                        else:

                            img = Image.open(my_pth)
                            img = g * np.array(img)
                            img = img[rmin:rmax, cmin:cmax]

                            img = Image.fromarray(img)

                            d = ImageDraw.Draw(img)
                            fpth = '/usr/share/fonts/truetype/abyssinica/AbyssinicaSIL-R.ttf'
                            fnt = ImageFont.truetype(fpth, fontsize)
                            d.rectangle((upleft, downright), fill=bgcolor)
                            d.text(upleft, str(val), font=fnt, fill=textcolor)

                            img = np.array(img)

                            image_1 = np.concatenate(
                                (image_1, img),
                                axis=1
                            )

    image_0 = np.concatenate((gt[rmin:rmax, cmin:cmax], image_0), axis=1)
    image_1 = np.concatenate((np.zeros_like(gt[rmin:rmax, cmin:cmax]), image_1), axis=1)

    image = np.concatenate((image_0, image_1), axis=0)
    image[(image[..., 0] == 0) * (image[..., 1] == 0) * (image[..., 2] == 0)] = 255


    image = Image.fromarray(image)

    x, y = image.size

    image = image.resize((x // 4, y // 4))

    image.save('{}_cat.png'.format(img_id))

