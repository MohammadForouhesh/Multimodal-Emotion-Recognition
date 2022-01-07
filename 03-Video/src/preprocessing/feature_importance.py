import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.multiclass import OneVsRestClassifier
from xgboost import XGBClassifier

path = '/Users/maelfabien/filrouge_pole_emploi/Video/'
local_path = '/Users/maelfabien/Desktop/LocalDB/Videos/'

X_train = np.load(local_path + "X_train.npy")
X_test = np.load(local_path + "X_test.npy")
y_train = np.load(local_path + "y_train.npy")
y_test = np.load(local_path + "y_test.npy")

shape_x = 48
shape_y = 48
nRows, nCols, nDims = X_train.shape[1:]
input_shape = (nRows, nCols, nDims)
classes = np.unique(y_train)
nClasses = len(classes)


model = OneVsRestClassifier(XGBClassifier())
model.fit(X_train.reshape(-1,48*48*1)[:5000], y_train[:5000])

gray = cv2.cvtColor(model.feature_importances_.reshape(shape_x, shape_y,3), cv2.COLOR_BGR2GRAY)
plt.figure(figsize=(12,8))
sns.heatmap(gray)
plt.show()
