import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split

filename = '../models/xgb_multi_class_model.sav'
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

# creating training and test sets
x_train, x_test, y_train, y_test = train_test_split(test_down_data, downTest.gammaMid.values, test_size=0.2,
                                                    random_state=4)

# load the model from disk
loaded_model = joblib.load(filename)
result = loaded_model.predict(x_test)
a = 0
print("The prediction for the test data down set is ", result)
for i in range(len(result)):
    if result[i] == 0:
        # print('match', y_pred[i])
        a = a + 1

print('y_pred length ', len(result))
print('number of matches ', a)
print('accuracy ', a / len(result), '%')
