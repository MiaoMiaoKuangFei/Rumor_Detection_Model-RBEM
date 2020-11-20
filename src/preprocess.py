#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2020/11/13 21:16
# @Author  : Zhutian Lin
# @FileName: preprocess.py
# @Software: PyCharm
import time
import jieba
import re
from zhon.hanzi import punctuation
import numpy as np
import pandas as pd

"""
Preprocess for dataset.
"""


class rumor_info:
    """
    Class contains rumor detection info containing title, summary, time and labels.
    """
    def __init__(self, timestamp, labels, cut_smry, cut_title):
        """
        Construct function.
        :param timestamp:
        :param labels:
        :param cut_smry:
        :param cut_title:
        """
        self.timestamp = timestamp
        self.labels = labels
        self.cut_smry = cut_smry
        self.cut_title = cut_title


def load_dataset(path):
    """
    Load dataset from disk to memory.
    :param path:str of path
    :return:Dataframe, loaded dataset
    """
    a = pd.read_csv(path).sort_values(['crawlTime'])
    return a


def cut_and_clear(sentence):
    """
    Jieba cut words and drop punctuation.
    :param sentence:raw sentence.
    :return:clear sentence list.
    """
    return list(jieba.cut(re.sub("[%s]+" % punctuation.__add__('...'), "", sentence)))  # 额外清掉…


def preprocess_rumor(rumors):
    """
    Convert the cut dataset to the format used one.
    :param rumors:rumor information.
    :return:
    """
    lid_map = {'fake': 0, 'true': 1, 'doubt': 2}
    labels = list(map(lambda x: lid_map.get(x), list(rumors['rumorType'])))
    crawlTime = list(rumors['crawlTime'])
    mainSummary = list(rumors['mainSummary'])
    title = list(rumors['title'])

    cut_title = list(map(lambda _: cut_and_clear(_), title))
    cut_smry = list(map(lambda _: cut_and_clear(_), mainSummary))
    timestamp = list(map(lambda x: int(time.mktime(time.strptime(x, "%Y-%m-%d"))), crawlTime))

    return rumor_info(timestamp, labels, cut_smry, cut_title)


def train_test_split(ratio, ri):
    """
    Train-test split process by ratio, which ratio=size(train)/size(all).
    :param ratio: ratio of training set to data set.
    :param ri: preprocess.rumor_info object contains all information to use.
    :return:
    """
    train_size = np.int(len(ri.timestamp) * ratio)
    train_ri = rumor_info(
        ri.timestamp[:train_size],
        ri.labels[:train_size],
        ri.cut_smry[:train_size],
        ri.cut_title[:train_size]
    )
    test_ri = rumor_info(
        ri.timestamp[train_size:],
        ri.labels[train_size:],
        ri.cut_smry[train_size:],
        ri.cut_title[train_size:]
    )
    return train_ri, test_ri


if __name__ == '__main__':
    print(preprocess_rumor(load_dataset("dataset/covid19_rumors.csv")))
