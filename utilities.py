# coding=utf-8
import os
import sys
import pickle
import argparse
from scipy import ndimage
import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical
from keras import backend as K

LETTER_LABELS = {
    'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8,
    'J': 9
}


def load_mnist(scaled=False):
    """Scale mnist data to range [0, 1] and change the label to
    binary categorical."""
    img_rows, img_cols = 28, 28
    num_classes = 10
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    if scaled:
        x_train /= 255
        x_test /= 255
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)
    return (x_train, y_train), (x_test, y_test)


def save_notmnist():
    """Load and normalize not mnist dataset."""
    dataset = np.ndarray(shape=(10000, 28, 28, 1), dtype=np.float32)
    data_labels = []
    data_dir = os.path.join(os.path.curdir, 'data', 'notMNIST')
    n = 0
    for label in LETTER_LABELS.keys():
        path = os.path.join(data_dir, label)
        files = os.listdir(path)
        for filename in files:
            image_path = os.path.join(path, filename)
            image_data = (ndimage.imread(image_path).astype(float))
            dataset[n, :, :, 0] = image_data
            data_labels.append(LETTER_LABELS[label])
            n += 1
    output_path = os.path.join(os.path.curdir, 'data', 'notMNIST.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump((dataset, np.array(data_labels)), f)


def load_notmnist(scaled=False):
    """Load not MNIST data."""
    input_path = os.path.join(os.path.curdir, 'data', 'notMNIST.pkl')
    with open(input_path, 'rb') as f:
        dataset, data_label = pickle.load(f)
    if scaled:
        dataset /= 255
    return dataset, data_label


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Doing some utilities')
    parser.add_argument(
        '-d', '--fun', help='The utilities function you want to run',
        required=True)
    args = vars(parser.parse_args())

    # Input directory
    if args['fun'].lower() == 'save_notmnist':
        save_notmnist()
    else:
        print 'Wrong -f args. Must be save_notmnist for now.'
        sys.exit()
