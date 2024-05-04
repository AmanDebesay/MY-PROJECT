import matplotlib.pyplot as plt
from mlxtend.data import loadlocal_mnist
import platform
from sklearn.linear_model import LogisticRegression
import tensorflow
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import utils
from tensorflow.keras import optimizers
from matplotlib import pyplot as plt
import torch

if not platform.system() == 'Windows':
    X_train, y_train = loadlocal_mnist(
        images_path='train-images-idx3-ubyte',
        labels_path='train-labels-idx1-ubyte')

else:
    X_train, y_train = loadlocal_mnist(
        images_path='train-images.idx3-ubyte',
        labels_path='train-labels.idx1-ubyte')
if not platform.system() == 'Windows':
    X_test, y_test = loadlocal_mnist(
        images_path='t10k-images.idx3-ubyte',
        labels_path='t10k-labels.idx1-ubyte')

else:
    X_test, y_test = loadlocal_mnist(
        images_path='t10k-images.idx3-ubyte',
        labels_path='t10k-labels.idx1-ubyte')
My_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

#DNN METHOD

# loop through each picture in the data set
if not platform.system() == 'Windows':
    X_train, y_train = loadlocal_mnist(
        images_path='train-images-idx3-ubyte',
        labels_path='train-labels-idx1-ubyte')
    test = X_test[1]
    test = test.reshape([32, 32])
    plt.gray()
    plt.imshow(test)
    plt.show()

# Set random seed for reproducability
tensorflow.random.set_seed(0)

print("Libraries imported.")
print('Keras version:', keras.__version__)
print('TensorFlow version:', tensorflow.__version__)

# Set data types for float features
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Set data types for categorical labels
y_train = utils.to_categorical(y_train)
y_test = utils.to_categorical(y_test)
print('Ready...')

# Define a classifier network
hl = 10  # Number of hidden layer nodes

model = keras.models.Sequential()
model.add(Dense(10, input_dim=784, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy')

#hyper-parameters for optimizer
learning_rate = 0.001
opt = optimizers.Adam(learning_rate)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

# Train the model over 10 epochs using 10-observation batches and using the test holdout dataset for validation
num_epochs = 10
history = model.fit(X_train, y_train, epochs=num_epochs, batch_size=10, validation_data=(X_test, y_test))
print(model.summary())

# Review training and validation
