import numpy as np
from glob import glob
import cv2
import os
import random 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split 

IMG_LIST = []
LABEL_LIST = []

def normalize(x):
    min_val = np.min(x)
    max_val = np.max(x)
    x = (x-min_val) / (max_val-min_val)
    return x

def next_batch(num, data, labels):
    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]
    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

def one_hot_encode(x, NROFCLASSES):
    encoded = np.zeros((len(x), NROFCLASSES))
    for idx, val in enumerate(x):
        encoded[idx][val] = 1
    return encoded

def read_train_valid_data(directory):
    for img_file in glob(directory + "/NSFW/*.png"):
        nsfw_label = img_file.split('/')[-2]
        img = cv2.imread(img_file)
        try:
            img = cv2.resize(img, (224, 224))
            IMG_LIST.append(img)
            LABEL_LIST.append(nsfw_label)
        except:
            print('Error at {}'.format(img_file))
    for img_file in glob(directory + "/SFW/*.png"):
        img = cv2.imread(img_file)
        sfw_label = img_file.split('/')[-2]
        try:
            img = cv2.resize(img, (224, 224))
            IMG_LIST.append(img)
            LABEL_LIST.append(sfw_label)
        except:
            print('Error at {}'.format(img_file))
    np.save('images_224.npy', IMG_LIST)
    np.save('labels_224.npy', LABEL_LIST)
    assert(len(IMG_LIST) == len(LABEL_LIST))

def train_test_split_data(NROFCLASSES):
    mapping_file = open("mapping_file_labels.txt", "w+")
    if os.path.isfile('images.npy') and os.path.isfile('labels.npy'):
        print('Numpy Arrays Exists!')
        IMAGES = np.load('images_224.npy')
        LABELS = np.load('labels_224.npy')
        unique_labels = list(set(LABELS))
        labelEncoder = preprocessing.LabelEncoder()
        labelEncoder.fit(unique_labels)
        mapping_file.write(str(unique_labels))
        mapping_file.write("\n")
        mapping_file.write(str(list(labelEncoder.transform(unique_labels))))
        mapping_file.close()
        labelEncoder.fit(LABELS)
        encoded_labels = labelEncoder.transform(LABELS)
        one_hot_labels = one_hot_encode(encoded_labels, NROFCLASSES)
        x_train, x_val, y_train, y_val = train_test_split(IMAGES, one_hot_labels, test_size=0.2, random_state=42)
        x_train = np.asarray(x_train)
        x_val = np.asarray(x_val)
        y_val = np.asarray(y_val)
        y_train = np.asarray(y_train)
        print("Done Data Formation.")
        print('X_TRAIN, Y_TRAIN Shape', x_train.shape, y_train.shape)
        print('X_VAL, Y_VAL Shape', x_val.shape, y_val.shape)
        return x_train, x_val, y_train, y_val
    else:
        print("Doesn't exist")
        read_train_valid_data("/Dataset/Train")
        print('Recall TRAIN_TEST_SPLIT_DATA() to get the data.')