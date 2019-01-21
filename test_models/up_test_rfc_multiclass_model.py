import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split

filename = '../models/rfc_multi_class_model.sav'
upTest = pd.read_csv("../data_files/up/upTest.csv")
# upTest = pd.read_csv("../data_files/down/downTest5Secs.csv")

test_down_data = np.array([[0 for x in range(9)] for y in range(len(upTest))])
for i in range(len(upTest)):
    test_down_data[i] = [upTest.eegRawValue.values[i],
                         upTest.delta.values[i],
                         upTest.theta.values[i],
                         upTest.alphaLow.values[i],
                         upTest.alphaHigh.values[i],
                         upTest.betaLow.values[i],
                         upTest.betaHigh.values[i],
                         upTest.gammaLow.values[i],
                         upTest.gammaMid.values[i]]

# creating training and test sets
x_train, x_test, y_train, y_test = train_test_split(test_down_data, upTest.gammaMid.values, test_size=0.2,
                                                    random_state=4)

# load the model from disk
loaded_model = joblib.load(filename)
result = loaded_model.predict(x_test)
a = 0
print("The prediction for the test data down set is ", result)
for i in range(len(result)):
    if result[i][0] == [0.] and result[i][1] == [0.] and result[i][2] == [1.]:
        # print('match', y_pred[i])
        a = a + 1

print('y_pred length ', len(result))
print('number of matches ', a)
print('accuracy ', a / len(result), '%')
