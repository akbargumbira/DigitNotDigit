# coding=utf-8
from keras.models import load_model
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix


class DigitNotDigit(object):
    """Class DigitNotDigit."""
    def __init__(self):
        """The constructor."""
        self._model = load_model('model/cnn_mnist_5_sigmoid.h5')
        self._threshold = 200

    def predict(self, data):
        """Predict the given data.

        :param data: The input data to predict
        :type data: list

        :return: The predicted label
        :rtype: list
        """
        predictions = self._model.predict(data)
        rest_mean = (np.sum(predictions, 1) - np.max(predictions, 1)) / 9
        rest_mean[rest_mean == 0] = 10**-10  # to avoid dividing by 0
        max_rest = np.max(predictions, 1) / rest_mean

        predicted_label = np.zeros(len(data))
        predicted_label[max_rest > self._threshold] = 1

        return predicted_label

    def evaluate_score(self, data, true_label):
        """Evaluate metrics for given data and its true label.

        :param data: The input data to predict.
        :type data: list

        :param true_label: The ground truth label.
        :type data: list

        :return: A tuple of accuracy and the confusion matrix
        :rtype: tuple
        """
        predicted_label = self.predict(data)
        accuracy = accuracy_score(true_label, predicted_label)
        conf = confusion_matrix(true_label, predicted_label)
        return accuracy, conf