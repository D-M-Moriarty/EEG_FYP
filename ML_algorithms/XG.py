# This script is forked from "XGBoost example (Python)" by DataCanary
# https://www.kaggle.com/datacanary/xgboost-example-python?scriptVersionId=108683
# here we used probability distribution for Ages instead of using mean or median
# because there are 263 missed values and filling them with list of values may increase accuracy

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the data
# train_df = pd.read_csv('../input/train.csv', header=0)
# test_df = pd.read_csv('../input/test.csv', header=0)

updf1 = pd.read_csv("../data_files/data_from_phone_recordings/up/upFor5(1).csv")
updf2 = pd.read_csv("../data_files/data_from_phone_recordings/up/upFor5(2).csv")
updf3 = pd.read_csv("../data_files/data_from_phone_recordings/up/upFor5(3).csv")
updf4 = pd.read_csv("../data_files/data_from_phone_recordings/up/upFor5(4).csv")
updf5 = pd.read_csv("../data_files/data_from_phone_recordings/up/upFor5(5).csv")
downdf1 = pd.read_csv("../data_files/data_from_phone_recordings/down/downFor5(1).csv")
downdf2 = pd.read_csv("../data_files/data_from_phone_recordings/down/downFor5(2).csv")
downdf3 = pd.read_csv("../data_files/data_from_phone_recordings/down/downFor5(3).csv")
downdf4 = pd.read_csv("../data_files/data_from_phone_recordings/down/downFor5(4).csv")
downdf5 = pd.read_csv("../data_files/data_from_phone_recordings/down/downFor5(5).csv")
relax_1_minute = pd.read_csv("../data_files/data_from_phone_recordings//relax/Relax1Minute.csv")

frames = [downdf1, downdf2, downdf3, downdf4,
          downdf5, updf1, updf2, updf3, updf4, updf5, relax_1_minute]
df = pd.concat(frames)

# create an array of shape 30706, 9 = number of records by the features
data = np.array([[0 for x in range(9)] for y in range(len(df))])
for i in range(len(df)):
    try:
        data[i] = [df.eegRawValue.values[i],
                   df.delta.values[i],
                   df.theta.values[i],
                   df.alphaLow.values[i],
                   df.alphaHigh.values[i],
                   df.betaLow.values[i],
                   df.betaHigh.values[i],
                   df.gammaLow.values[i],
                   df.gammaMid.values[i]]
    except:
        print(df.action.values[i])

encoder = LabelEncoder()
labels = encoder.fit_transform(df.action.values)
print(labels[0])
print(df.action.values[0])
print(labels[36000])
print(df.action.values[36000])
print(labels[16000])
print(df.action.values[16000])

# creating training and test sets
x_train, x_test, y_train, y_test = train_test_split(data, labels)
gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05).fit(x_train, y_train)
y_pred = gbm.predict(x_test)
print(y_pred)
print("The score for XGBoost ", gbm.score(x_test, y_test))

# confusion matrix
from sklearn.metrics import confusion_matrix
from sklearn import metrics

cm = confusion_matrix(y_test, y_pred)
print(cm)
# Model Accuracy, how often is the classifier correct?
print(len(y_train))
print("Accuracy for x_test:", metrics.accuracy_score(y_test, y_pred))

downTest = pd.read_csv("../data_files/data_from_phone_recordings/down/downTest5Secs.csv")
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
# print(encoder.inverse_transform(y_test[0]))
y_pred = gbm.predict(test_down_data)
a = 0
# print("The prediction for the test data down set is ", y_pred)
for i in range(len(y_pred)):
    if y_pred[i] == [0]:
        # print('match', y_pred[i])
        a = a + 1

# print('y_pred length ', len(y_pred))
# print('number of matches ', a)
print('accuracy ', a / len(y_pred), '%')

# joblib_file = "../models/xgb_multi_class_model.sav"
# joblib.dump(gbm, joblib_file)
