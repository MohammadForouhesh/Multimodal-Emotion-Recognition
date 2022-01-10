from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.multiclass import OneVsRestClassifier


def data_loader(path='../../Dataset/'):
    X_train = np.load(path + "X_train.npy")
    X_test = np.load(path + "X_test.npy")
    y_train = np.load(path + "y_train.npy")
    y_test = np.load(path + "y_test.npy")
    return X_train, X_test, y_train, y_test


def xgb_classifier(X_train, X_test, y_train, y_test):
    shape_x = 48
    shape_y = 48

    model = OneVsRestClassifier(XGBClassifier())
    model.fit(X_train.reshape(-1, 48*48*1)[:5000], y_train[:5000])

    gray = cv2.cvtColor(model.feature_importances_.reshape(shape_x, shape_y, 3), cv2.COLOR_BGR2GRAY)
    plt.figure(figsize=(12, 8))
    sns.heatmap(gray)
    plt.show()

    return model


if __name__ == '__main__':
    X = data_loader()
    xgb_classifier(*X)