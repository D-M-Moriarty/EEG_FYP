import numpy as np
import pandas as pd

downdf1 = pd.read_csv("../data_files/down/downFor5(1).csv")
downdf2 = pd.read_csv("../data_files/down/downFor5(2).csv")
downdf3 = pd.read_csv("../data_files/down/downFor5(3).csv")
downdf4 = pd.read_csv("../data_files/down/downFor5(4).csv")
downdf5 = pd.read_csv("../data_files/down/downFor5(5).csv")

frames = [downdf1, downdf2, downdf3, downdf4, downdf5]
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

data_image = []

for i in range(len(data)):
    for j in range(9):
        data_image.append(data[i][j])


# Create a function called "chunks" with two arguments, l and n:
def chunks(l, n):
    # For item i in a range that is a length of l,
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i + n]


list = list(chunks(data_image, 90))

a = np.asarray(list)

outfile = '../data_files/down_images/down.csv'
outfptr = open(outfile, 'a');

for i in range(len(a)):
    outfptr.write('down,');
    for j in range(len(a[i])):
        outfptr.write(str(a[i][j]) + ',')
    outfptr.write('\n');
