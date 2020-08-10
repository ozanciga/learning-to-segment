import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import glob
import os
from PIL import Image
import re

criteria = [
    's_acc',
    's_acc_fg',
    's_iou',
    's_bach',
    's_f1mic',
    's_f1mac',
]

df = pd.read_csv('exp_results.csv', delimiter=',')

plot_pts = []

'read in gt'
gt_paths = 'data/val/gt_thumbnails/'
gt_image_coeff = {}
for gt_pth in sorted(glob.glob('{}/*.png'.format(gt_paths))):
    filename = os.path.basename(gt_pth)
    filename = re.findall('(.+).png', filename)[0]

    gt = Image.open(gt_pth)
    gt_image_coeff[filename] = gt.size[0] * gt.size[1]

gt_image_coeff_sum = 0
for key in gt_image_coeff:
    gt_image_coeff_sum += gt_image_coeff[key]
gt_image_coeff = {k: v / gt_image_coeff_sum for k, v in gt_image_coeff.items()}

percent_labels = ['0%', '1%', '2.5%', '5%', '7.5%', '10%', '15%', '20%', '25%', '30%', '40%', '50%', '75%', '100%']
percent_seg = [0., 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1]
exp_iters = [1, 2, 3, 4, 5]
seg_and_clss = [0, 1, 2]

for use_seg_ratio in tqdm(percent_seg):
    for seg_and_cls in seg_and_clss:
        max_score_exp_iter_all = {k: len(exp_iters)*[0] for k in criteria}
        for exp_iter in exp_iters:
            for epoch in range(0, 0 + 2):
                stats = df[(df.exp_iter == exp_iter)
                           & (df.seg_and_cls == seg_and_cls)
                           & (df.use_seg_ratio == use_seg_ratio)
                           & (df.epoch == epoch)
                           & (df.case_id != 'A08')]
                if len(stats) < 1:
                    continue
                for k in criteria:
                    # stats[k] = stats[k] * list(gt_image_coeff.values())[3:]
                    avgs = np.mean(np.array(stats[k]))
                    if avgs > max_score_exp_iter_all[k][exp_iter-1]:
                        max_score_exp_iter_all[k][exp_iter-1] = avgs
        #if np.any(np.array(list((max_score_exp_iter_all.values()))).flatten()):
        plot_pts.append((seg_and_cls, use_seg_ratio, max_score_exp_iter_all))

for k in criteria:
    subp = [(it[0], it[1], np.mean(it[2][k]), np.std(it[2][k])) for it in plot_pts]

    p = np.array(subp)
    p_cls = p[p[:, 0] == 1][:, 1:]
    p_seg = p[p[:, 0] == 0][:, 1:]
    p_ccs = p[p[:, 0] == 2][:, 1:]  # cls (full dataset) + seg

    #p_ccs[:, 1] /= p_seg[:, 1][-1]
    #p_cls[:, 1] /= p_seg[:, 1][-1]
    #p_seg[:, 1] /= p_seg[:, 1][-1]

    p_cls = p_ccs
    print('bach', k)
    for j in range(p_cls.shape[0]):
        end_or_amp = ' \\\ ' if j == p_cls.shape[0] - 1 else '&'
        print(f' {100 * p_cls[j, 1]:.1f} $\pm$ {100 * p_cls[j, 2]:.1f} {end_or_amp} ', end='')
    print('')

    fig, ax = plt.subplots()

    plt.plot(p_cls[:, 0], p_cls[:, 1], 'bx-', label='S+C')
    plt.fill_between(
        p_cls[:, 0],
        p_cls[:, 1] - p_cls[:, 2],
        p_cls[:, 1] + p_cls[:, 2],
        alpha=0.5,
        edgecolor='#1B2ACC',
        facecolor='#089FFF',
        linewidth=0
    )

    plt.plot(p_seg[:, 0], p_seg[:, 1], 'rx-', label='S')
    plt.fill_between(
        p_seg[:, 0],
        p_seg[:, 1] - p_seg[:, 2],
        p_seg[:, 1] + p_seg[:, 2],
        alpha=0.5,
        edgecolor='#CC4F1B',
        facecolor='#FF9848',
        linewidth=0
    )

    plt.plot(p_ccs[:, 0], p_ccs[:, 1], 'kx--', label='S+C*')
    plt.fill_between(
        p_ccs[:, 0],
        p_ccs[:, 1] - p_ccs[:, 2],
        p_ccs[:, 1] + p_ccs[:, 2],
        alpha=0.5,
        edgecolor='#808080',
        facecolor='#D3D3D3',
        linewidth=0
    )

    #plt.xlabel('% of seg. images')
    #plt.ylabel('Accuracy')
    #plt.title('Comparison btw only seg. vs seg+cls {}'.format(k))
    plt.legend(loc='lower right')
    plt.grid()
    plt.xticks(percent_seg[::2], percent_labels[::2], rotation='vertical')
    plt.show()
    fig.savefig('exp_plots/exp_results_{}.pdf'.format(k), dpi=fig.dpi, bbox_inches='tight')
    fig.savefig('exp_plots/pngs/exp_results_{}.png'.format(k), dpi=fig.dpi, bbox_inches='tight')
