import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import glob
import os
from PIL import Image
import re

cls_show_ids = [1, 2, 3]

criteria = [
    's_cfs',
]

df = pd.read_csv('exp_results_cfs.csv', delimiter=',')

plot_pts = []

percent_labels = ['0%', '1%', '2.5%', '5%', '7.5%', '10%', '15%', '20%', '25%', '30%', '40%', '50%', '75%', '100%']
percent_seg = [0., 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1]
exp_iters = [1, 2, 3, 4, 5]
seg_and_clss = [0, 1, 2]

for use_seg_ratio in tqdm(percent_seg):
    for seg_and_cls in seg_and_clss:
        max_score_exp_iter_all = {k: len(exp_iters)*[np.array([0, 0, 0, 0])] for k in criteria}
        for exp_iter in exp_iters:
            for epoch in range(0, 1 + 1):
                stats = df[(df.exp_iter == exp_iter)
                           & (df.seg_and_cls >= seg_and_cls)
                           & (df.use_seg_ratio == use_seg_ratio)
                           & (df.epoch == epoch)]

                if len(stats) < 1:
                    continue
                for k in criteria:
                    avgs = []
                    for xx in stats[k].values:
                        avgs.append([float(item) for item in xx.replace('[', '').replace(']', '').split(', ') if item != ''])
                    avgs = np.array(avgs)
                    avgs = avgs.mean(0)
                    if avgs[1:].mean() > max_score_exp_iter_all[k][exp_iter-1][1:].mean():
                        max_score_exp_iter_all[k][exp_iter-1] = avgs
        plot_pts.append((seg_and_cls, use_seg_ratio, max_score_exp_iter_all))

for k in criteria:

    subp = [(it[0], it[1], np.mean(it[2][k], axis=0), np.std(it[2][k], axis=0)) for it in plot_pts]

    p = np.array(subp)
    p_cls = p[p[:, 0] == 1][:, 1:]
    p_seg = p[p[:, 0] == 0][:, 1:]
    p_ccs = p[p[:, 0] == 2][:, 1:]  # cls (full dataset) + seg

    cls_color = ['k', 'r', 'g', 'b']

    for cls_id in cls_show_ids:

        fig, ax = plt.subplots()

        p_cls_sub = np.array([(it[0], it[1][cls_id], it[2][cls_id]) for it in p_cls])
        p_seg_sub = np.array([(it[0], it[1][cls_id], it[2][cls_id]) for it in p_seg])
        p_ccs_sub = np.array([(it[0], it[1][cls_id], it[2][cls_id]) for it in p_ccs])

        p_cls_sub[:, 1] /= p_seg_sub[:, 1][-1]
        p_ccs_sub[:, 1] /= p_seg_sub[:, 1][-1]
        p_seg_sub[:, 1] /= p_seg_sub[:, 1][-1]

        plt.plot(p_cls_sub[:, 0], p_cls_sub[:, 1], '{}o-'.format(cls_color[cls_id]), label='S+C')
        plt.fill_between(
            p_cls_sub[:, 0],
            p_cls_sub[:, 1] - p_cls_sub[:, 2],
            p_cls_sub[:, 1] + p_cls_sub[:, 2],
            alpha=0.5,
            edgecolor='#1B2ACC',
            facecolor='#089FFF',
            linewidth=0
        )

        plt.plot(p_seg_sub[:, 0], p_seg_sub[:, 1], '{}x--'.format(cls_color[cls_id]), label='S')
        plt.fill_between(
            p_seg_sub[:, 0],
            p_seg_sub[:, 1] - p_seg_sub[:, 2],
            p_seg_sub[:, 1] + p_seg_sub[:, 2],
            alpha=0.5,
            edgecolor='#CC4F1B',
            facecolor='#FF9848',
            linewidth=0
        )

        plt.plot(p_ccs_sub[:, 0], p_ccs_sub[:, 1], '{}x:'.format(cls_color[cls_id]), label='S+C*')
        plt.fill_between(
            p_ccs_sub[:, 0],
            p_ccs_sub[:, 1] - p_ccs_sub[:, 2],
            p_ccs_sub[:, 1] + p_ccs_sub[:, 2],
            alpha=0.5,
            edgecolor='#808080',
            facecolor='#D3D3D3',
            linewidth=0
        )

        plt.legend(loc='lower right')
        plt.grid()
        plt.xticks(percent_seg[::2], percent_labels[::2], rotation='vertical')
        plt.show()

        fig.savefig('exp_plots/exp_results_cfs_{}_{}.pdf'.format(k, cls_id), dpi=fig.dpi, bbox_inches='tight')
        fig.savefig('exp_plots/pngs/exp_results_cfs_{}_{}.png'.format(k, cls_id), dpi=fig.dpi, bbox_inches='tight')
