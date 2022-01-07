import numpy as np
import pandas as pd
from keras.utils import to_categorical
from matplotlib import pyplot as plt

path = '/Users/maelfabien/filrouge_pole_emploi/Video/'
local_path = '/Users/maelfabien/Desktop/LocalDB/Videos/'

pd.options.mode.chained_assignment = None  # default='warn'  #to suppress SettingWithCopyWarning

dataset = pd.read_csv(local_path + 'fer2013.csv')

train = dataset[dataset["Usage"] == "Training"]

test = dataset[dataset["Usage"] == "PublicTest"]

train['pixels'] = train['pixels'].apply(lambda image_px : np.fromstring(image_px, sep = ' '))
test['pixels'] = test['pixels'].apply(lambda image_px : np.fromstring(image_px, sep = ' '))
dataset['pixels'] = dataset['pixels'].apply(lambda image_px : np.fromstring(image_px, sep = ' '))

shape_x = 48
shape_y = 48


X_train = train.iloc[:, 1].values
y_train = train.iloc[:, 0].values

X_test = test.iloc[:, 1].values
y_test = test.iloc[:, 0].values

X = dataset.iloc[:,1].values
y = dataset.iloc[:,0].values

X_train = np.vstack(X_train)
X_test = np.vstack(X_test)
X = np.vstack(X)

X_train = np.reshape(X_train, (X_train.shape[0], 48, 48, 1))
y_train = np.reshape(y_train, (y_train.shape[0], 1))

X_test = np.reshape(X_test, (X_test.shape[0], 48, 48, 1))
y_test = np.reshape(y_test, (y_test.shape[0], 1))

X = np.reshape(X, (X.shape[0], 48, 48, 1))
y = np.reshape(y, (y.shape[0], 1))

print("Shape of X_train and y_train is " + str(X_train.shape) + " and " + str(y_train.shape) + " respectively.")
print("Shape of X_test and y_test is " + str(X_test.shape) + " and " + str(y_test.shape) + " respectively.")


X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X = X.astype('float32')

X_train /= 255
X_test /= 255
X /= 255

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y = to_categorical(y)

classes = np.unique(y_train)
nClasses = len(classes)
print('Total number of outputs : ', nClasses)
print('Output classes : ', classes)

# Find the shape of input images and create the variable input_shape
nRows, nCols, nDims = X_train.shape[1:]
input_shape = (nRows, nCols, nDims)


def get_label(argument):
    labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
    return labels.get(argument, "Invalid emotion")

plt.figure(figsize=[10,5])

# Display the first image in training data
plt.subplot(121)
plt.imshow(np.squeeze(X_train[25,:,:], axis = 2), cmap='gray')
plt.title("Ground Truth : {}".format(get_label(int(y_train[0]))))

# Display the first image in testing data
plt.subplot(122)
plt.imshow(np.squeeze(X_test[26,:,:], axis = 2), cmap='gray')
plt.title("Ground Truth : {}".format(get_label(int(y_test[1500]))))


np.save(local_path + 'X_train', X_train)
np.save(local_path + 'X_test', X_test)
np.save(local_path + 'X', X)
np.save(local_path + 'y_train', y_train)
np.save(local_path + 'y_test', y_test)
np.save(local_path + 'y', y)