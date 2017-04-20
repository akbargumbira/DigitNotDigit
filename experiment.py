from __future__ import print_function
import os
import keras
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import cv2
import numpy as np
import matplotlib.pyplot as plt
from digitnotdigit import DigitNotDigit
from utilities import load_notmnist


def print_stats(predictions):
    print('Max of max', np.max(np.max(predictions, 1)))
    print('Min of max', np.min(np.max(predictions, 1)))
    print('Average of max', np.average(np.max(predictions, 1)))
    print('Average of min', np.average(np.min(predictions, 1)))
    print('Median of max', np.median(np.max(predictions, 1)))
    print('Percentile 95 of max', np.percentile(np.max(predictions, 1), 95))
    print('# max > 0.04', np.where(np.max(predictions, 1) > 0.04)[0].size)
    print('# max > 0.03', np.where(np.max(predictions, 1) > 0.03)[0].size)
    print('# max > 0.02', np.where(np.max(predictions, 1) > 0.02)[0].size)
    print('# max > 0.01', np.where(np.max(predictions, 1) > 0.01)[0].size)
    print('Average of average', np.average(np.average(predictions, 1)))
    print('Max of average', np.max(np.average(predictions, 1)))
    print('Min of average', np.min(np.average(predictions, 1)))
    print('# avg > 0.004', np.where(np.average(predictions, 1) > 0.004)[0].size)
    print('# avg > 0.003', np.where(np.average(predictions, 1) > 0.003)[0].size)
    print('# avg > 0.002', np.where(np.average(predictions, 1) > 0.002)[0].size)
    print('# avg > 0.001', np.where(np.average(predictions, 1) > 0.001)[0].size)
    rest = (np.sum(predictions, 1) - np.max(predictions, 1)) / 9
    rest[rest == 0] = 0.000000000001
    max_rest = np.max(predictions,1) / rest
    print('Average of max/(sum-max)', np.average(max_rest))
    print('Max of max/(sum-max)', np.max(max_rest))
    print('Min of max/(sum-max)', np.min(max_rest))
    print('# max/(sum-max) > 10', np.where(max_rest > 10)[0].size)
    print('# max/(sum-max) > 70', np.where(max_rest > 70)[0].size)
    print('# max/(sum-max) > 100', np.where(max_rest > 100)[0].size)
    print('# max/(sum-max) > 150', np.where(max_rest > 150)[0].size)
    print('# max/(sum-max) > 200', np.where(max_rest > 200)[0].size)
    print('# max/(sum-max) > 300', np.where(max_rest > 300)[0].size)
    print('# max/(sum-max) > 500', np.where(max_rest > 500)[0].size)
    print('# max/(sum-max) > 1000', np.where(max_rest > 1000)[0].size)

# batch_size = 128
# num_classes = 10
# epochs = 5
# #
# # # input image dimensions
# img_rows, img_cols = 28, 28
#
# # the data, shuffled and split between train and test sets (60k training, 10k test)
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# cv2.imshow('tes', x_train[0:10].reshape(28*10, 28))
# cv2.waitKey(0)
#
# if K.image_data_format() == 'channels_first':
#     x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
#     x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
#     input_shape = (1, img_rows, img_cols)
# else:
#     x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
#     x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
#     input_shape = (img_rows, img_cols, 1)
#
# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
# x_train /= 255
# x_test /= 255
# print('x_train shape:', x_train.shape)
# print(x_train.shape[0], 'train samples')
# print(x_test.shape[0], 'test samples')
#
# # convert class vectors to binary class matrices
# y_train = keras.utils.to_categorical(y_train, num_classes)
# y_test = keras.utils.to_categorical(y_test, num_classes)
#
# # # Training
# model = Sequential()
# model.add(
#     Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes, activation='sigmoid'))
#
# model.compile(loss=keras.losses.categorical_crossentropy,
#               optimizer=keras.optimizers.Adadelta(),
#               metrics=['accuracy'])
#
# model.fit(x_train, y_train,
#           batch_size=batch_size,
#           epochs=epochs,
#           verbose=1,
#           validation_data=(x_test, y_test))

# # Save trained model
# model.save(os.path.join(os.path.curdir, 'model', 'cnn_mnist_5.h5'))
# model.save(os.path.join(os.path.curdir, 'model', 'cnn_mnist_5_sigmoid.h5'))

# Experiment with random image 10k, 28 * 28 images
# Generate random test
# n_random_image = 10000
# random_test_data = np.random.rand(n_random_image, 28, 28, 1)
#
# # Load model
# model = load_model('model/cnn_mnist_5_sigmoid.h5')
#
# print('Predictions of training data 60k..............')
# predicts_training = model.predict(x_train)
# print_stats(predicts_training)
# # plt.plot(np.arange(len(x_train)), np.max(predicts_training, 1), 'ro')
# # plt.show()
# # plt.plot(np.arange(len(x_train)), np.average(predicts_training, 1), 'ro')
# # plt.show()
# print('Predictions of test data 10k..............')
# predicts_test = model.predict(x_test)
# print_stats(predicts_test)
# # plt.plot(np.arange(len(x_test)), np.max(predicts_test, 1), 'ro')
# # plt.show()
# # plt.plot(np.arange(len(x_test)), np.average(predicts_test, 1), 'ro')
# # plt.show()
# print('Predictions of random test data 10k...........')
# predicts_random = model.predict(random_test_data)
# print_stats(predicts_random)
# # plt.plot(np.arange(n_random_image), np.max(predicts_random, 1), 'ro')
# # plt.show()
# # plt.plot(np.arange(n_random_image), np.average(predicts_random, 1), 'ro')
# # plt.show()
#
#
# # Show random images that have max > 0.05
# # random_might_be_digit = random_test_data[
# #     np.where(np.max(predicts_random, 1) > 0.04)[0]]
# # random_might_be_digit *= 255
# # cv2.imshow('tes', random_might_be_digit.reshape(28*len(random_might_be_digit), 28))
# # cv2.waitKey(0)
#
# #
# # # x_train = random_test_data * 255
# # # cv2.imshow('tes', np.concatenate(
# # #     (x_train[0], x_train[1], x_train[2], x_train[3], x_train[4],
# # #      x_train[5], x_train[6], x_train[7], x_train[8], x_train[9]),
# # #     axis=1))
# # # cv2.waitKey(0)


classifier = DigitNotDigit()
notmnist_dataset, notmnist_label = load_notmnist(normalized=True)
