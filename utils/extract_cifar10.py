import numpy as np
import pickle
import cv2
import os 

def unpickle(filepath):
    with open(filepath, 'rb') as fo:
        dict = pickle.load(fo, encoding = 'latin-1')
    return dict
    
def load_cifar_pickle(filepath):
    dict = unpickle(filepath)
    images = dict['data']
    images = np.reshape(images, (10000, 3, 32, 32))
    labels = np.array(dict['labels'])
    print("Loaded {} labelled images.".format(images.shape[0]))
    return images, labels 

def load_cifar_categories(filepath):
    dict = unpickle(filepath)
    return dict['label_names']

def save_cifar_image(array, path):
    # array is 3x32x32. cv2 needs 32x32x3
    array = array.transpose(1,2,0)
    # array is RGB. cv2 needs BGR
    array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    # save to PNG file
    return cv2.imwrite(path, array)
