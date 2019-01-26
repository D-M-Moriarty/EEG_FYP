import numpy as np
import pandas as pd

relax_1_minute = pd.read_csv("../data_files/relax/Relax1Minute.csv")

# create an array of shape 30706, 9 = number of records by the features
data = np.array([[0 for x in range(9)] for y in range(len(relax_1_minute))])
for i in range(len(relax_1_minute)):
    try:
        data[i] = [relax_1_minute.eegRawValue.values[i],
                   relax_1_minute.delta.values[i],
                   relax_1_minute.theta.values[i],
                   relax_1_minute.alphaLow.values[i],
                   relax_1_minute.alphaHigh.values[i],
                   relax_1_minute.betaLow.values[i],
                   relax_1_minute.betaHigh.values[i],
                   relax_1_minute.gammaLow.values[i],
                   relax_1_minute.gammaMid.values[i]]
    except:
        print(relax_1_minute.action.values[i])

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

outfile = '../data_files/rest_images/rest.csv'
outfptr = open(outfile, 'a');

for i in range(len(a)):
    outfptr.write('rest,');
    for j in range(len(a[i])):
        outfptr.write(str(a[i][j]) + ',')
    outfptr.write('\n');
