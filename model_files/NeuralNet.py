import numpy as np
import pandas as pd
from keras.metrics import categorical_accuracy
from keras.optimizers import Adam

updf1 = pd.read_csv("../data_files/up/upFor5(1).csv")
updf2 = pd.read_csv("../data_files/up/upFor5(2).csv")
updf3 = pd.read_csv("../data_files/up/upFor5(3).csv")
updf4 = pd.read_csv("../data_files/up/upFor5(4).csv")
updf5 = pd.read_csv("../data_files/up/upFor5(5).csv")
downdf1 = pd.read_csv("../data_files/down/downFor5(1).csv")
downdf2 = pd.read_csv("../data_files/down/downFor5(2).csv")
downdf3 = pd.read_csv("../data_files/down/downFor5(3).csv")
downdf4 = pd.read_csv("../data_files/down/downFor5(4).csv")
downdf5 = pd.read_csv("../data_files/down/downFor5(5).csv")
frames = [downdf1, downdf2, downdf3, downdf4, downdf5, updf1, updf2, updf3, updf4, updf5]
df = pd.concat(frames)

downTest = pd.read_csv("../data_files/down/downTest5Secs.csv")
test_down_data = np.array([[0 for x in range(9)] for y in range(len(downTest))])
for i in range(len(downTest)):
    test_down_data[i] = [downTest.eegRawValue.values[i],
                         downTest.delta.values[i],
                         downTest.theta.values[i],
                         downTest.alphaLow.values[i],
                         downTest.alphaHigh.values[i],
                         downTest.betaLow.values[i],
                         downTest.betaHigh.values[i],
                         downTest.gammaLow.values[i],
                         downTest.gammaMid.values[i]]
downTest2 = pd.read_csv("../data_files/down/downTest2.csv")
test_down_data2 = np.array([[0 for x in range(9)] for y in range(len(downTest2))])
for i in range(len(downTest2)):
    test_down_data2[i] = [downTest.eegRawValue.values[i],
                          downTest.delta.values[i],
                          downTest.theta.values[i],
                          downTest.alphaLow.values[i],
                          downTest.alphaHigh.values[i],
                          downTest.betaLow.values[i],
                          downTest.betaHigh.values[i],
                          downTest.gammaLow.values[i],
                          downTest.gammaMid.values[i]]
upTest = pd.read_csv("../data_files/up/upTest.csv")
test_up = np.array([[0 for x in range(9)] for y in range(len(upTest))])
for i in range(len(upTest)):
    test_up[i] = [upTest.eegRawValue.values[i],
                  upTest.delta.values[i],
                  upTest.theta.values[i],
                  upTest.alphaLow.values[i],
                  upTest.alphaHigh.values[i],
                  upTest.betaLow.values[i],
                  upTest.betaHigh.values[i],
                  upTest.gammaLow.values[i],
                  upTest.gammaMid.values[i]]

# create an array of shape 30706, 9 = number of records by the features
data = np.array([[0 for x in range(9)] for y in range(len(df))])
for i in range(len(df)):
    data[i] = [df.eegRawValue.values[i],
               df.delta.values[i],
               df.theta.values[i],
               df.alphaLow.values[i],
               df.alphaHigh.values[i],
               df.betaLow.values[i],
               df.betaHigh.values[i],
               df.gammaLow.values[i],
               df.gammaMid.values[i]]

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

encoder = LabelBinarizer()
labels = encoder.fit_transform(df.action.values)
print(labels.shape)

# creating training and test sets
x_train, x_test, y_train, y_test = train_test_split(data, labels)

from keras import models
from keras import layers

network = models.Sequential()
network.add(layers.Dense(64, input_shape=(9,)))
network.add(layers.Dense(32, activation="relu"))
network.add(layers.Dense(1, activation='sigmoid'))

# Adam = Adam(lr=0.05)
network.compile(optimizer="Adam",
                loss='binary_crossentropy',
                metrics=['acc'])

history = network.fit(x_train, y_train,
                      epochs=10, verbose=1)

loss_and_metrics = network.evaluate(x_test, y_test)
print('loss and metrics', loss_and_metrics)

print('prediction: ', network.predict(x_test))

import matplotlib.pyplot as plt

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
# network = models.Sequential()
# network.add(layers.Dense(64, input_shape=(9,)))
# network.add(layers.Dense(32, activation="relu"))
# network.add(layers.Dense(1, activation='sigmoid'))
#
# network.compile(optimizer='Adam',
#                 loss='binary_crossentropy',
#                 metrics=['acc'])
#
# print(network.fit(x_train, y_train, epochs=5))
#
# print("The prediction ", network.predict(test_down_data))
