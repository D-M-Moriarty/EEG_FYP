import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from xgboost import XGBClassifier

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

encoder = LabelBinarizer()
labels = encoder.fit_transform(df.action.values)

print(labels[0])
print(df.action.values[0])
print(labels[16000])
print(df.action.values[16000])
# creating training and test sets
x_train, x_test, y_train, y_test = train_test_split(data, labels)
# XGBoost
xgb = XGBClassifier()
xgb.fit(x_train, y_train)
print("The score for XGBoost ", xgb.score(x_test, y_test))
y_pred = xgb.predict(x_test)
print("The prediction for the test set is ", y_pred)
print(encoder.inverse_transform(y_test[0]))
print("The prediction for the test data down set is ", xgb.predict(test_down_data))
test_down_data_labels = encoder.fit_transform(downTest.action.values)
print(encoder.inverse_transform(test_down_data_labels[0]))
print("The prediction for the test data down2 set is ", xgb.predict(test_down_data2))

print("XGBoost most important features ", xgb.feature_importances_)

import pandas as pd
feature_imp = pd.Series(xgb.feature_importances_).sort_values(ascending=False)
feature_imp
import matplotlib.pyplot as plt
import seaborn as sns

# Creating a bar plot
sns.barplot(x=feature_imp, y=feature_imp.index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features from XGBoost")
plt.legend()
plt.show()