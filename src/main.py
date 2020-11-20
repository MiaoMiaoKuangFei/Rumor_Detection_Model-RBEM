#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2020/11/13 22:31
# @Author  : Zhutian Lin
# @FileName: main.py
# @Software: PyCharm
import src.preprocess as sp
import src.trainer as sr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

'''
Entrance for serial experiments
'''

if __name__ == '__main__':
    #  Data pre-processing.
    print("\t\t\t------------------------------------------------\t\t\t")
    print("\t\t\t\t\tData Preprocessing\t\t\t")
    print("\t\t\t------------------------------------------------\t\t\t")
    whole_ds = sp.preprocess_rumor(sp.load_dataset("dataset/covid19_rumors.csv"))
    train_ri, test_ri = sp.train_test_split(0.5, whole_ds)
    trt, trs, tet, tes = sr.tokenlize_train_and_test(train_ri, test_ri)

    #  Decision Tree Classifier.
    print("\t\t\t------------------------------------------------\t\t\t")
    print("\t\t\t\t\tDecisionTreeClassifier Training\t\t\t")
    print("\t\t\t------------------------------------------------\t\t\t")

    train_y = train_ri.labels.copy()
    test_y = test_ri.labels.copy()

    clf_t = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=3)
    clf_t.fit(trt, train_y)
    clf_t_predict = clf_t.predict(tet)
    print("Only Title with Tree Classifier(Accuracy):")
    print(accuracy_score(test_y, clf_t_predict))
    x3 = clf_t.predict(trt)

    clf_s = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=3)
    clf_s.fit(trs, train_y)
    clf_s_predict = clf_s.predict(tes)
    print("Only Summary with Tree Classifier(Accuracy):")
    print(accuracy_score(test_y, clf_t_predict))
    x4 = clf_s.predict(trs)

    print("\t\t\t------------------------------------------------\t\t\t")
    print("\t\t\t\t\tDNN Training\t\t\t")
    print("\t\t\t------------------------------------------------\t\t\t")

    # for title
    th_title, model_title = sr.trainer(train_x=trt, train_y=train_ri.labels)
    print("Only Title with RNN-Trainer(accuracy):")
    pt = sr.judge(model=model_title, test_x=tet, test_y=test_ri.labels)
    # # for summary
    th_summary, model_summary = sr.trainer(train_x=trs, train_y=train_ri.labels)
    print("Only Summary with RNN-Trainer(accuracy):")
    ps = sr.judge(model=model_summary, test_x=tes, test_y=test_ri.labels)

    #  Use LR as ensemble method.
    x1 = model_title.predict_classes(trt)
    x2 = model_summary.predict_classes(trs)
    ensemble_model = LogisticRegression()

    esb_x = [[xx1[0], xx2[0]] for (xx1, xx2) in zip(x1, x2)]

    ensemble_model.fit(esb_x, train_ri.labels)

    print("Ensemble Accuracy for RBEM:")
    esb_tx = [[xx1[0], xx2[0]] for (xx1, xx2) in zip(pt, ps)]
    y_pred = ensemble_model.predict(esb_tx)
    print(accuracy_score(test_ri.labels, y_pred))
