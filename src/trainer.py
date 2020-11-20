#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2020/11/13 21:52
# @Author  : Zhutian Lin
# @FileName: trainer.py
# @Software: PyCharm

from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import SimpleRNN
import numpy as np

"""
Separate RNN-trainer.
"""


def tokenlize_train_and_test(train_ri, test_ri):
    """
    Tokenlizing process for train and test
    :param train_ri: preprocess.rumor_info object with train data, including summary, title and time.
    :param test_ri: preprocess.rumor_info object with test data, including summary, title and time.
    :return:
    """
    tokenizer = Tokenizer(num_words=4095)
    text = []

    text.extend(train_ri.cut_title)
    text.extend(train_ri.cut_smry)
    text.extend(test_ri.cut_title)
    text.extend(test_ri.cut_smry)

    tokenizer.fit_on_texts(text)

    x_train_title_ids = tokenizer.texts_to_sequences(train_ri.cut_title)
    x_train_smry_ids = tokenizer.texts_to_sequences(train_ri.cut_smry)
    x_test_title_ids = tokenizer.texts_to_sequences(test_ri.cut_title)
    x_test_smry_ids = tokenizer.texts_to_sequences(test_ri.cut_smry)

    x_train_title = sequence.pad_sequences(x_train_title_ids, maxlen=32)
    x_train_smry = sequence.pad_sequences(x_train_smry_ids, maxlen=32)
    x_test_title = sequence.pad_sequences(x_test_title_ids, maxlen=32)
    x_test_smry = sequence.pad_sequences(x_test_smry_ids, maxlen=32)

    return x_train_title, x_train_smry, x_test_title, x_test_smry


def trainer(train_x, train_y):
    """
    RNN-trainer.
    :param train_x: train x list.
    :param train_y: train y list.
    :return: train history and model, used in downstream task.
    """
    if type(train_x) is not np.ndarray:
        train_x = np.array(train_x)
    if type(train_y) is not np.ndarray:
        train_y = np.array(train_y)

    model = Sequential()
    model.add(Embedding(output_dim=64, input_dim=4096, input_length=32))
    model.add(SimpleRNN(units=128, activation='swish', dropout=0.1))
    model.add(Dense(units=1, activation='swish'))
    model.build()
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])

    train_history = model.fit(train_x, train_y, epochs=10)

    return train_history, model


def judge(model, test_x, test_y):
    """
    Judge single Simple-RNN trainer result.
    :param model:
    :param test_x:
    :param test_y:
    :return:
    """
    if type(test_x) is not np.ndarray:
        test_x = np.array(test_x)
    if type(test_y) is not np.ndarray:
        test_y = np.array(test_y)
    scores = model.evaluate(test_x, test_y, verbose=0)
    print(scores)

    return model.predict_classes(test_x)
