#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2020/11/10 16:27
# @Author  : Zhutian Lin
# @FileName: data_distribution.py
# @Software: PyCharm
import os
from matplotlib import font_manager as fm
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib import rcParams
import src.preprocess as sp
import sys

'''
Count labels and make histogram
'''

if __name__ == '__main__':

    # input arguments.
    ds_path = sys.argv[1]
    font_path = sys.argv[2]

    rumors = sp.load_dataset(ds_path)
    rumors_cnt = dict(Counter(rumors['rumorType']))

    fpath = os.path.join(rcParams["datapath"], font_path)
    prop = fm.FontProperties(fname=fpath)  # set font

    x = [_[0] for _ in rumors_cnt.items()]
    y = [_[1] for _ in rumors_cnt.items()]

    plt.xlabel("Label of Statement", fontproperties=prop)
    plt.ylabel("Count", fontproperties=prop)
    plt.title("Label Distribution", fontproperties=prop)

    plt.bar(x, y)

    for a, b in zip(x, y):  # histogram values display
        plt.text(a, b, '%.2d' % b, ha='center', va='bottom', fontsize=7, fontproperties=prop)

    # save image.
    plt.savefig("./figure/distribution.png")
    plt.show()
