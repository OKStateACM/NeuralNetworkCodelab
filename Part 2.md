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

## Writing the Neural Network

Now comes the fun part! Now that we have the data, we can start constructing a neural network. We will use a python library called **keras** to do this.

Keras is a high level api for creating neural networks. It abstracts away a lot of the details.

### Layers

Keras has a concept called **Layers** that represent each layer in a neural network. Remember that each layer is a collection of neurons. For our purposes, we want to use a **Dense Layer**. We can use one like this

```python
Dense(units=32, activation='relu')
```

This creates a dense layer with 32 neurons that uses a relu activation function

### Model

We stack multiple layers together in a **model**. Add this to your code

```python
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam

def create_model():
    model = Sequential()

    model.add(Dense(units=100, activation='relu', input_dim=784)) # hidden layer
    model.add(Dense(units=10, activation='softmax')) # output layer

    return model

model = create_model()
```

This creates a model that has one hidden layer and one output layer. The hidden layer has 100 units and uses a relu activation function. The output layer has 10 outputs (1 for each possible digit) and uses the softmax activation function. The softmax activation function makes it so that the probabilities in the vector add to 1.

### Compiling and Training the Model

Now that we've created the model, we need to **compile** it, this just tells the model how it is going to be trained as well as a few other things. Add this to your code

```python
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
```

The ```loss='categorical_crossentropy'``` bit tells the model that the way it will measure loss is by using the categorical_crossentropy function. Don't worry too much about what this is specifically, just know that it is a loss function that measures how wrong the predictions are.

The ```optimizer=Adam(lr=0.001)``` bit just tells the model to use the Adam optimizer with a learning rate of 0.001. Again, don't worry too much about the specifics, just know that the optimizer will update the parameters of the model to minimize the loss

### Training the Model

Ok, so we now have the data and we have a model. Let's train the model using the data! Add this to your code

```python
model.fit(x_train, y_train)
```

And finally, let's write some code to evaluate how we did on our test set

```python
print(model.evaluate(x_test, y_test))
```

## The final code

Your final code should look something like this

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
```

Run it and see what happens!

[Part 3: Where to Go From Here]()
