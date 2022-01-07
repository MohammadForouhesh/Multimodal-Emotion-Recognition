
from __future__ import division

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier

### CNN models ###


path = '/Users/maelfabien/filrouge_pole_emploi/Video/'
local_path = '/Users/maelfabien/Desktop/LocalDB/Videos/'


X_train = np.load(local_path + "X_train.npy")
X_test = np.load(local_path + "X_test.npy")
y_train = np.load(local_path + "y_train.npy")
y_test = np.load(local_path + "y_test.npy")


shape_x = 48
shape_y = 48
nRows,nCols,nDims = X_train.shape[1:]
input_shape = (nRows, nCols, nDims)
classes = np.unique(y_train)
nClasses = len(classes)

clf = OneVsRestClassifier(LGBMClassifier(learning_rate = 0.1, num_leaves = 10, n_estimators=10, verbose=1))
clf.fit(X_train.reshape(-1,48*48*1), y_train)

y_pred = clf.predict(X_test.reshape(-1,48*48*1))
print(accuracy_score(y_pred, y_test))

