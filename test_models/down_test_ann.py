import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

filename = '../models/neural_net.h5'
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

upTest = pd.read_csv("../data_files/up/upTest.csv")

test_up_data = np.array([[0 for x in range(9)] for y in range(len(upTest))])
for i in range(len(upTest)):
    test_up_data[i] = [upTest.eegRawValue.values[i],
                       upTest.delta.values[i],
                       upTest.theta.values[i],
                       upTest.alphaLow.values[i],
                       upTest.alphaHigh.values[i],
                       upTest.betaLow.values[i],
                       upTest.betaHigh.values[i],
                       upTest.gammaLow.values[i],
                       upTest.gammaMid.values[i]]

encoder = LabelBinarizer()
labels = encoder.fit_transform(downTest.action.values)

# creating training and test sets
x_train, x_test, y_train, y_test = train_test_split(test_down_data, labels, test_size=0.2,
                                                    random_state=4)

X_std = (x_train - x_train.min(axis=0)) / (x_train.max(axis=0) - x_train.min(axis=0))
X_scaled = X_std * (np.max(x_train) - np.min(x_train)) + np.min(x_train)

from keras import models

model = models.load_model(filename)

model.summary()

# evaluate loaded model on test data
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
down_pred = model.predict(test_down_data[0:])
up_pred = model.predict(test_up_data[0:9])

print("The prediction for the test data down set is ", down_pred)
print("The prediction for the test data up set is ", up_pred)
a = 0
for i in range(len(down_pred)):
    if down_pred[i][0] == [1.] and down_pred[i][1] == [0.] and down_pred[i][2] == [0.]:
        # print('match', y_pred[i])
        a = a + 1

print('y_pred length ', len(down_pred))
print('number of matches ', a)
print('accuracy ', a / len(down_pred), '%')

a = 0
for i in range(len(up_pred)):
    if up_pred[i][0] == [0.] and up_pred[i][1] == [0.] and up_pred[i][2] == [1.]:
        # print('match', y_pred[i])
        a = a + 1

print('y_pred length ', len(up_pred))
print('number of matches ', a)
print('accuracy ', a / len(up_pred), '%')
# score = model.evaluate(X_scaled, y_train, verbose=0)
# print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
