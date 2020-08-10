from tensorflow.python.summary.summary_iterator import summary_iterator
import glob
import os
import json
import numpy as np
import argparse
from argparse import Namespace
import matplotlib.pyplot as plt

def parse_events_file(path: str) -> dict:
    # ref: https://stackoverflow.com/a/55822721
    metrics = {}
    for e in summary_iterator(path):
        for v in e.summary.value:
            if isinstance(v.simple_value, float):
                if v.tag not in metrics:
                    metrics[v.tag] = []
                metrics[v.tag].append(v.simple_value)
    return metrics


dataset_name = 'digestpath2019_123'

paths = f'/home/osha/Desktop/ld2/y-net/exp_plots/seg/experiments/experiments/{dataset_name}/**/*events*'
paths = f'/home/osha/PycharmProjects/y-net/exp_plots/seg/experiments/experiments/{dataset_name}/**/*events*'

for metric in ['accuracy']:#, 'f1_micro', 'f1_macro']:

    results = {}

    o = {}

    for path in glob.glob(paths, recursive=1):

        metrics = parse_events_file(path)
        if metrics == {}:
            continue

        with open(f'{os.path.split(os.path.split(path)[0])[0]}/config.txt', "r") as f:
            parser = argparse.ArgumentParser()
            text = f.read().replace('array', '')
            args = parser.parse_args(namespace=eval(text))

        # pick the metric you want to evaluate
        key = f'Metrics/val/{metric}'
        if key not in metrics:
            max_acc = 0.
            continue
        else:
            m = metrics[key]
            max_acc = np.max(m)

        if args.seg_and_cls not in o:
            o[args.seg_and_cls] = {}

        if args.use_seg_ratio not in o[args.seg_and_cls]:
            o[args.seg_and_cls][args.use_seg_ratio] = []

        o[args.seg_and_cls][args.use_seg_ratio].append(max_acc)

    percent_labels = ['0%', '1%', '2.5%', '5%', '7.5%', '10%', '15%', '20%', '25%', '30%', '40%', '50%', '75%', '100%']
    percent_seg = [0.0, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0]

    cls_tasks = True
    normalize_to_one = False

    if cls_tasks:
        percent_labels = percent_labels[:-2]
        percent_seg = percent_seg[:-2]

    subp = []
    for percent in percent_seg:
        for mode in [1, 2, 3]:  # seg_and_cls
            if mode in o and percent in o[mode]:
                exp_iter_results = o[mode][percent]
                subp.append(
                    (mode, percent, np.mean(exp_iter_results, axis=0), np.std(exp_iter_results, axis=0))
                )

    p = np.array(subp)
    p_cls = p[p[:, 0] == 1][:, 1:]
    if not cls_tasks:
        p_seg = p[p[:, 0] == 0][:, 1:]
    p_ccs = p[p[:, 0] == 2][:, 1:]  # cls (full dataset) + seg
    if cls_tasks:
        p_ssc = p[p[:, 0] == 3][:, 1:]  # cls + seg (full dataset)

    fig, ax = plt.subplots()

    p_list = [p_ssc, p_cls, p_ccs] if cls_tasks else [p_seg, p_cls, p_ccs]
    print(dataset_name, metric)
    for p_current in p_list:
        for j in range(p_current.shape[0]):
            end_or_amp = ' \\\ ' if j == p_current.shape[0] - 1 else '&'
            print(f' {100*p_current[j, 1]:.1f} $\pm$ {100*p_current[j, 2]:.1f} {end_or_amp} ', end='')
        print('')

    if normalize_to_one:
        p_ccs[:, 1] /= p_cls[:, 1][-1]
        if not cls_tasks:
            p_seg[:, 1] /= p_cls[:, 1][-1]
        if cls_tasks:
            p_ssc[:, 1] /= p_cls[:, 1][-1]
        p_cls[:, 1] /= p_cls[:, 1][-1]

    plt.plot(p_cls[:, 0], p_cls[:, 1], 'bx-', label='S+C' if not cls_tasks else '$S_2+C_2$')
    plt.fill_between(
        p_cls[:, 0],
        p_cls[:, 1] - p_cls[:, 2],
        p_cls[:, 1] + p_cls[:, 2],
        alpha=0.5,
        edgecolor='#1B2ACC',
        facecolor='#089FFF',
        linewidth=0
    )

    if not cls_tasks:
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

    plt.plot(p_ccs[:, 0], p_ccs[:, 1], 'kx--', label='S+C*' if not cls_tasks else '$S_2+C*_2$')
    plt.fill_between(
        p_ccs[:, 0],
        p_ccs[:, 1] - p_ccs[:, 2],
        p_ccs[:, 1] + p_ccs[:, 2],
        alpha=0.5,
        edgecolor='#808080',
        facecolor='#D3D3D3',
        linewidth=0
    )

    if cls_tasks:
        plt.plot(p_ssc[:, 0], p_ssc[:, 1], 'mx--', label='S*+C' if not cls_tasks else '$S*_2+C_2$')
        plt.fill_between(
            p_ssc[:, 0],
            p_ssc[:, 1] - p_ssc[:, 2],
            p_ssc[:, 1] + p_ssc[:, 2],
            alpha=0.5,
            edgecolor='#404040',
            facecolor='pink',
            linewidth=0
        )

    plt.legend(loc='lower right')
    plt.grid()
    plt.xticks(percent_seg[::2], percent_labels[::2], rotation='vertical')
    plt.show()

    fig.savefig(f'{dataset_name}_{key.replace("Metrics/val/", "")}_{"normalized" if  normalize_to_one else "unnormalized"}.pdf', dpi=fig.dpi, bbox_inches='tight')

    fig.savefig(f'{dataset_name}_{key.replace("Metrics/val/", "")}_{"normalized" if  normalize_to_one else "unnormalized"}.png', dpi=fig.dpi, bbox_inches='tight')
