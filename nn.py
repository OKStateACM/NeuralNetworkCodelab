from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape([-1, x_train.shape[1] * x_train.shape[2]])
x_test = x_test.reshape([-1, x_test.shape[1]*x_test.shape[2]])

x_train = x_train/255.
x_test = x_test/255.

y_train = np.eye(10)[y_train]
y_test = np.eye(10)[y_test]
