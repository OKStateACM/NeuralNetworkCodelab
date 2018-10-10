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

from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam

def create_model():
    model = Sequential()

    model.add(Dense(units=100, activation='relu', input_dim=784)) # hidden layer
    model.add(Dense(units=10, activation='softmax')) # output layer

    return model

model = create_model()
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
model.fit(x_train, y_train)
print(model.evaluate(x_test, y_test))

print('Done')
