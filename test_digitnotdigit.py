# coding=utf-8
import unittest
import numpy as np
from keras.datasets import mnist
from digitnotdigit import DigitNotDigit
from utilities import normalized_mnist


class TestDigitNotDigit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.random_test_data = np.random.rand(10000, 28, 28, 1)
        (cls.x_train, cls.y_train), (cls.x_test, cls.y_test) = \
            normalized_mnist()

    def setUp(self):
        self.classifier = DigitNotDigit()

    def test_predict(self):
        # Predict 10k random image, Label with 1 should be less than 500
        predicted_label = self.classifier.predict(self.random_test_data)
        self.assertLess(sum(predicted_label), 500)

        # Predict with MNIST, label with 1 should be greater than 9500
        predicted_label = self.classifier.predict(self.x_test)
        self.assertGreater(sum(predicted_label), 9500)

    def test_evaluate_score(self):
        # random data
        acc, conf = self.classifier.evaluate_score(
            self.random_test_data,
            np.zeros(len(self.random_test_data)))
        self.assertGreater(acc, 0.98)
        self.assertEqual(conf[1][0], 0)
        self.assertEqual(conf[1][1], 0)

        # mnist test 10k
        acc, conf = self.classifier.evaluate_score(
            self.x_test,
            np.ones(len(self.x_test)))
        self.assertGreater(acc, 0.95)
        self.assertEqual(conf[0][0], 0)
        self.assertEqual(conf[0][1], 0)



