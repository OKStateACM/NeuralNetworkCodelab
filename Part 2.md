# Part 2: Using a Neural Network in code

Now comes the fun part, we're going to be using a high level framework, Keras, to build a neural network that will recognize handwritten digits

First, let's get Keras installed. It's just a python package, so installing it with Pip should be a breeze
```bash
pip install keras
```

If everything ran smoothly, you should be able to run the ```python``` command to enter a python shell, type in ```import keras``` and get no errors. Type ```quit()``` to exit the python shell

Let's also install ```matplotlib```, which will be helpful for visualizing things

```bash
pip install matplotlib
```

## Getting the data

The data we'll be using for training and testing comes from the [MNIST Database](https://en.wikipedia.org/wiki/MNIST_database), a large database of handwritten digits that is commonly used in machine learning examples. You can think of classifying handwritten digits as the "Hello World" of machine learning. Since it's so common, Keras actually makes it easy to use this dataset. Create a new file called ```nn.py``` and insert this into the file.

```python
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train[0])
```

Run the program

```bash
python nn.py
```

What do you see? You should see a huge 2 dimensional array that doesn't make much sense. Let's take a better look at it. Replace the code with this
```python
from keras.datasets import mnist
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()

plt.imshow(x_train[0], cmap='gray')
plt.show()
```
Now you should see a handwritten 5, let's verify that the y value for this is a 5. Add this to the end of your code
```python
print(y_train[0])
```

See! The x matches with the y. Feel free to try this out with other values.

But let's take a closer look at x_train and y_train. They're both **numpy arrays**, if you're not familiar with numpy arrays, they're basically just multidimensional arrays. Let's look at the shape of these.

```python
print('x_train shape:', x_train.shape)
print('y_train_shape', y_train.shape)
```

Let's focus on x_train first. You can see it's shape is (60000, 28, 28). This means it has 60000 28x28 arrays. So there are 60000 training examples and each example is a 28x28 array representing an image. Remember from the theory lesson though that we want each example to be one vector, so let's roll each example out to be a 784 dimensional vector. python makes this easy

```python
x_train = x_train.reshape([-1, x_train.shape[1] * x_train.shape[2]])

print('x_train shape after reshaping:', x_train.shape)
```

Now each example should be a 784 dimensional vector, perfect! We do also want to make this data range from 0-1 instead of 0-255

```python
x_train = x_train/255.
print(x_train[0])
```

You should now see that the values in x_train[0] range from 0-1

## Processing the Labels

Now we need to process the labels. Right now, they're just ints. We need them to be probability vectors of floats. For example, if the label is a 5, the probability vector will be
```
[0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0]
```

Here's a neat trick to do this. ```np.eye(n)``` creates a n by n identity matrix. So ```np.eye(3)``` will produce
```
1 0 0
0 1 0
0 0 1
```

If we have a list (or np array) ```a = [0 2 1 0 2 1]``` and we index ```np.eye(3)``` with it likeso: ```np.eye(3)[a]```, we'll get:

```
[[1,0,0], [0,0,1], [0,1,0], [1,0,0], [0,1,0], [0, 1, 0]]
```

So if our y_train is a bunch of int indexes and we want to turn that into probability vectors (these are called **one hot vectors**), we can do something similar. Add this to your code
```python
import numpy as np
y_train = np.eye(10)[y_train]
print(y_train[0])
```

Now each element of y is a one hot vector, just like what we wanted. After removing all the print statements and doing the same preprocessing to the test data, our code should look like this:

```python
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
```
