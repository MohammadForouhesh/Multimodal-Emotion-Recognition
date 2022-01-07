from time import time

import cv2
import dlib
import numpy as np
import scipy
from matplotlib import pyplot as plt
from skimage.feature import hog
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

pictures = dataset['pixels']
labels = dataset['emotion']

# %%

shape_x = 48
shape_y = 48
window_size = 24
window_step = 6

ONE_HOT_ENCODING = False
SAVE_IMAGES = False
GET_LANDMARKS = False
GET_HOG_FEATURES = False
GET_HOG_WINDOWS_FEATURES = False
SELECTED_LABELS = []
IMAGES_PER_LABEL = 500

OUTPUT_FOLDER_NAME = "/Users/maelfabien/Desktop/LocalDB/Videos/Face_Features/"

# %%

predictor = dlib.shape_predictor(
    '/Users/maelfabien/Desktop/LocalDB/Videos/landmarks/shape_predictor_68_face_landmarks.dat')


# %%

def get_landmarks(image, rects):
    if len(rects) > 1:
        raise BaseException("TooManyFaces")
    if len(rects) == 0:
        raise BaseException("NoFaces")
    return np.matrix([[p.x, p.y] for p in predictor(image, rects[0]).parts()])


# %%

def sliding_hog_windows(image):
    hog_windows = []
    for y in range(0, shape_x, window_step):
        for x in range(0, shape_y, window_step):
            window = image[y:y + window_size, x:x + window_size]
            hog_windows.extend(hog(window, orientations=8, pixels_per_cell=(8, 8),
                                   cells_per_block=(1, 1), visualise=False))
    return hog_windows


# %%

images = []
labels_list = []
landmarks = []

hog_slide_features = []
hog_slide_images = []

hog_features = []
hog_images = []

for i in range(len(pictures)):
    try:
        # Build the image as an array
        image = pictures[i].reshape((shape_x, shape_y))
        images.append(pictures[i])

        # HOG sliding windows features
        features = sliding_hog_windows(image)
        f, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualise=True)
        hog_slide_features.append(features)
        hog_slide_images.append(hog_image)

        # HOG features
        features, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                                  cells_per_block=(1, 1), visualise=True)
        hog_features.append(features)
        hog_images.append(hog_image)

        # Facial landmarks
        scipy.misc.imsave('temp.jpg', image)
        image2 = cv2.imread('temp.jpg')
        face_rects = [dlib.rectangle(left=1, top=1, right=47, bottom=47)]
        face_landmarks = get_landmarks(image2, face_rects)
        landmarks.append(face_landmarks)

        # Labels
        labels_list.append(labels[i])
        # nb_images_per_label[labels[i]] += 1

    except Exception as e:
        print("error in image: " + str(i) + " - " + str(e))

np.save(OUTPUT_FOLDER_NAME + 'labels.npy', labels_list)

np.save(OUTPUT_FOLDER_NAME + 'hog_slide_image.npy', hog_slide_images)
np.save(OUTPUT_FOLDER_NAME + 'hog_slide_features.npy', hog_slide_features)

np.save(OUTPUT_FOLDER_NAME + 'hog_image.npy', hog_images)
np.save(OUTPUT_FOLDER_NAME + 'hog_features.npy', hog_features)

np.save(OUTPUT_FOLDER_NAME + 'landmarks.npy', landmarks)
np.save(OUTPUT_FOLDER_NAME + 'images.npy', images)

labels_list = np.load(OUTPUT_FOLDER_NAME + 'labels.npy')

hog_slide_images = np.load(OUTPUT_FOLDER_NAME + 'hog_slide_image.npy')
hog_slide_features = np.load(OUTPUT_FOLDER_NAME + 'hog_slide_features.npy')

hog_images = np.load(OUTPUT_FOLDER_NAME + 'hog_image.npy')
hog_features = np.load(OUTPUT_FOLDER_NAME + 'hog_features.npy')

landmarks = np.load(OUTPUT_FOLDER_NAME + 'landmarks.npy')
images = np.load(OUTPUT_FOLDER_NAME + 'images.npy')


for i in range(10):
    fig, axs = plt.subplots(nrows=1, ncols=2, sharex=True)
    ax = axs[0]
    ax.imshow(images[i].reshape((shape_x, shape_y)))
    ax.set_title('Face')

    ax = axs[1]
    ax.imshow(hog_images[i])
    ax.set_title('HOG image')


landmarks = np.array([x.flatten() for x in landmarks])
landmarks.shape


data_0 = hog_features
data_1 = landmarks
data_2 = np.concatenate((landmarks, hog_features), axis=1)
data_3 = np.concatenate((landmarks, hog_slide_features), axis=1)

data_0.shape, data_1.shape, data_2.shape, data_3.shape


X_train, X_test, y_train, y_test = train_test_split(data_0, labels_list, test_size=0.25, random_state=42)


model = OneVsRestClassifier(SVC(random_state=42, max_iter=10000, kernel='rbf', gamma='auto'))

# Train
start_time = time()
model.fit(X_train, y_train)
training_time = time() - start_time
print("Training time : ", training_time)

y_pred = model.predict(X_test)
accuracy_hog = accuracy_score(y_pred, y_test)
print("Accuracy : ", accuracy_hog)


X_train, X_test, y_train, y_test = train_test_split(data_1, labels_list, test_size=0.25, random_state=42)


model = OneVsRestClassifier(SVC(random_state=42, max_iter=10000, kernel='rbf', gamma='auto'))


start_time = time()
model.fit(X_train, y_train)
training_time = time() - start_time
print("Training time : ", training_time)


y_pred = model.predict(X_test)
accuracy_hog = accuracy_score(y_pred, y_test)
print("Accuracy : ", accuracy_hog)


X_train, X_test, y_train, y_test = train_test_split(data_2, labels_list, test_size=0.25, random_state=42)

model = OneVsRestClassifier(SVC(random_state=42, max_iter=10000, kernel='rbf', gamma='auto'))


start_time = time()
model.fit(X_train, y_train)
training_time = time() - start_time
print("Training time : ", training_time)


y_pred = model.predict(X_test)
accuracy_hog = accuracy_score(y_pred, y_test)
print("Accuracy : ", accuracy_hog)

X_train, X_test, y_train, y_test = train_test_split(data_3, labels_list, test_size=0.25, random_state=42)


model = OneVsRestClassifier(SVC(random_state=42, max_iter=100, kernel='rbf', gamma='auto'))

start_time = time()
model.fit(X_train, y_train)
training_time = time() - start_time
print("Training time : ", training_time)

y_pred = model.predict(X_test)
accuracy_hog = accuracy_score(y_pred, y_test)
print("Accuracy : ", accuracy_hog)