#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2020/11/10 16:27
# @Author  : Zhutian Lin
# @FileName: data_distribution.py
# @Software: PyCharm
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt


def load_dataset(path):
    return pd.read_csv(path)


if __name__ == '__main__':
    rumors = load_dataset("dataset/covid19_rumors.csv")
    rumors_cnt = dict(Counter(rumors['rumorType']))
    x = [_[0] for _ in rumors_cnt.items()]
    y = [_[1] for _ in rumors_cnt.items()]
    plt.xlabel("Label of Statement")
    plt.ylabel("Count")
    plt.bar(x, y)
    plt.savefig("./figure/distribution.png")
    plt.show()

