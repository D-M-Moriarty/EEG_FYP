import pandas as pd
import numpy as np
from xgboost import XGBClassifier

df = pd.read_csv("../data_files/up/testEeg.csv")
# create an array of shape 30706, 9 = number of records by the features
train_data = np.array([[0 for x in range(9)] for y in range(len(df))])
for i in range(len(df)):
    train_data[i] = [df.eegRawValue.values[i],
                     df.delta.values[i],
                     df.theta.values[i],
                     df.alphaLow.values[i],
                     df.alphaHigh.values[i],
                     df.betaLow.values[i],
                     df.betaHigh.values[i],
                     df.gammaLow.values[i],
                     df.gammaMid.values[i]]

print("train data element 0 ", train_data[0])
print("label 0 ", df.action.values)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

encoder = LabelBinarizer()
train_labels = encoder.fit_transform(df.action.values)
train_labels.shape

x_train, x_test, y_train, y_test = train_test_split(train_data, train_labels)

# Random Forrest
rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)
print("The score for Random Forest ", rfc.score(x_test, y_test))
print("Random Forest most important features ", rfc.feature_importances_)

# XGBoost
xgb = XGBClassifier()
xgb.fit(x_train, y_train)
print("The score for XGBoost ", xgb.score(x_test, y_test))
print("XGBoost most important features ", xgb.feature_importances_)

from sklearn.preprocessing import LabelBinarizer
from keras.utils import to_categorical

encoder = LabelBinarizer()

train_labels = encoder.fit_transform(df.action.values)
train_labels.shape
# # train_labels = train_labels.T
# train_labels = encoder.fit_transform(train_labels)

# # train_labels = train_labels.astype('float32')
train_labels

from keras import models
from keras import layers

network = models.Sequential()
network.add(layers.Dense(64, input_shape=(9,)))
network.add(layers.Dense(32, activation="relu"))
network.add(layers.Dense(1, activation='sigmoid'))

network.compile(optimizer='Adam',
                loss='binary_crossentropy',
                metrics=['acc'])

print(network.fit(x_train, y_train, epochs=5))

print(network.predict(x_test))

network = models.Sequential()
network.add(layers.Dense(64, input_shape=(9,)))
network.add(layers.Dense(32, activation="relu"))
network.add(layers.Dense(1, activation='sigmoid'))

network.compile(optimizer='Adam',
                loss='binary_crossentropy',
                metrics=['acc'])

print(network.fit(x_train, y_train, epochs=5))

print("The prediction ", network.predict(x_test))
