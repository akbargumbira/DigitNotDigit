# coding=utf-8
import os
import argparse
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from utilities import load_mnist


def train(epochs=5):
    """Train the CNN and save the model"""
    batch_size = 128
    num_classes = 10
    input_shape = (1, 28, 28) if K.image_data_format() == 'channels_first' else (28, 28, 1)

    # Load MNIST dataset and normalized
    (x_train, y_train), (x_test, y_test) = load_mnist(scaled=True)

    # Training
    model = Sequential()
    model.add(
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='sigmoid'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))

    # Save trained model
    filename = 'cnn_mnist_%s_sigmoid.h5' % epochs
    model.save(os.path.join(os.path.curdir, 'model', filename))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train the network.')
    parser.add_argument(
        '-e', '--epochs', help='The number of epochs', required=False)
    args = vars(parser.parse_args())

    # Input directory
    if args['epochs']:
        epochs = int(args['epochs'])
        train(epochs)
    else:
        train()
