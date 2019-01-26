# ensure the plots are inside the notebook, not an external window

import numpy as np
import pandas as pd

up = pd.read_csv("../data_files/up_images/up.csv", header=None).as_matrix()
down = pd.read_csv("../data_files/down_images/down.csv", header=None).as_matrix()
rest = pd.read_csv("../data_files/rest_images/rest.csv", header=None).as_matrix()
mnist = pd.read_csv("../data_files/mnist_test.csv")

training_data_file = open("../data_files/up_images/up.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

data = up[:, 1:-1]
labels = up[:, 0]

data2 = down[:, 1:-1]
labels2 = down[:, 0]

data3 = rest[:, 1:-1]
labels3 = rest[:, 0]

data = np.append(data, data2, axis=0)
data = np.append(data, data3, axis=0)
labels = np.append(labels, labels2, axis=0)
labels = np.append(labels, labels3, axis=0)

print(data)
print(labels)

# # create an array of shape 30706, 9 = number of records by the features
# data = np.array([[0 for x in range(9)] for y in range(len(df))])
# for i in range(len(df)):
#     data[i] = [df.eegRawValue.values[i],
#                df.delta.values[i],
#                df.theta.values[i],
#                df.alphaLow.values[i],
#                df.alphaHigh.values[i],
#                df.betaLow.values[i],
#                df.betaHigh.values[i],
#                df.gammaLow.values[i],
#                df.gammaMid.values[i]]
#
#
#
# data_image = []
#
# for i in range(len(data)):
#     for j in range(9):
#         data_image.append(data[i][j])
#
# print(len(data_image))
# print(data_image)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

encoder = LabelBinarizer()
labels = encoder.fit_transform(labels)
print(labels)

# creating training and test sets
x_train, x_test, y_train, y_test = train_test_split(data, labels)

from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(90, 10, 1)))
# model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((3, 3)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.summary()

model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(3, activation='softmax'))

# add model layers
# model.add(layers.Conv2D(64, kernel_size=3, activation='relu', input_shape=(9, 10, 1)))
# model.add(layers.Conv2D(32, kernel_size=3, activation='relu'))
# model.add(layers.Flatten())
# model.add(layers.Dense(3, activation='softmax'))

model.summary()

x_train = x_train.reshape((423, 90, 10, 1))
print(np.max(x_train))
print(np.max(x_test))
print(np.min(x_train))
X_std = (x_train - x_train.min(axis=0)) / (x_train.max(axis=0) - x_train.min(axis=0))
X_scaled = X_std * (np.max(x_train) - np.min(x_train)) + np.min(x_train)

x_train = x_train.astype('float32') / X_scaled

# x_test = x_test.reshape((2, 90, 10, 1))
# x_test = x_test.astype('float32') / X_scaled

# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

print(y_train)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=15, batch_size=64)
