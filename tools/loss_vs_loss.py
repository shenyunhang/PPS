#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import cv2
import os
import sys
import re

from detectron.datasets.json_dataset import JsonDataset
from detectron.utils.io import load_object
import detectron.utils.vis as vis_utils

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

import matplotlib.pyplot as plt
import matplotlib
import sys
import subprocess
import numpy as np

print(sys.argv)

SNAPSHOT_EPOCHS = 5
LOG_PERIOD = 20

font = {'family': 'Arial', 'weight': 'normal', 'size': '18'}
matplotlib.rc('font', **font)


def get_loss(log_path):
    log_file = open(log_path, "r")
    prefix_path = os.path.splitext(log_path)[0]

    loss_values = []
    iter_values = []
    snapshot_values = []

    it = 0
    for line in log_file.readlines():
        line = line.strip()

        ma = re.search('model_final\.pkl', line)
        if ma is not None:
            break

        ma = re.search('model_epoch[0-9]+[.]pkl', line)
        if ma is not None:
            ma = re.search('[0-9]+', ma.group())
            snapshot_value = float(ma.group())
            snapshot_values.append(snapshot_value)

        ma = re.search('"loss": "[0-9]+([.][0-9]+)?"', line)
        if ma is None:
            continue
        # print(ma)
        # print(ma.group())
        # print(ma.groups())

        ma = re.search('[0-9]+([.][0-9]+)?', ma.group())
        loss_value = float(ma.group())
        loss_values.append(loss_value)

        iter_values.append(it)
        it += LOG_PERIOD

    return loss_values, iter_values, snapshot_values


def draw():
    log_path = sys.argv[1]
    output_dir = os.path.join(os.path.dirname(log_path), 'draw')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    colors = ['r', 'g', 'b', 'p']
    labels = ['w/o CRM', 'w/ CRM']

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    lines = []
    for log_path, color, label in zip(sys.argv[1:], colors, labels):
        print(log_path)
        loss_values, iter_values, snapshot_values = get_loss(log_path)

        # plot the data
        line, = ax1.plot(
            iter_values[::1],
            loss_values[::1],
            color,
            linewidth=0.5,
            label=label)
        lines.append(line)
    ax1.legend(handles=lines, prop={'size': 18})

    ax1.set_xlabel('Iterations', fontsize=22)
    ax1.set_ylabel('Loss', fontsize=22)

    # major_ticks = np.arange(1, iter_values[-1], 3000)
    # minor_ticks = np.arange(0, mAP_epoch_values[-1], 1)
    # ax1.set_xticks(major_ticks)
    # ax1.set_xticks(minor_ticks, minor=True)
    ax1.grid(which='both')

    # set the limits
    ax1.set_xlim([0, iter_values[-1]])
    ax1.set_ylim([0, loss_values[0]])
    ax1.set_ylim([0, 5000])
    fig.set_tight_layout(True)
    fig.set_size_inches(10., 5.)
    plt.savefig(os.path.join(output_dir, 'loss_vs_loss_plot.png'), dpi=100)


draw()
