import numpy as np
import pandas as pd

# Initializing the arrays required to store the data.
from notebook.notebookapp import raw_input

attention_values = np.array([])
meditation_values = np.array([])
delta_values = np.array([])
theta_values = np.array([])
lowAlpha_values = np.array([])
highAlpha_values = np.array([])
lowBeta_values = np.array([])
highBeta_values = np.array([])
lowGamma_values = np.array([])
highGamma_values = np.array([])
blinkStrength_values = np.array([])
time_array = np.array([])

import sys
import json
import time
from telnetlib import Telnet

tn = Telnet('localhost', 13854);

start = time.clock();

i = 0;
# app registration step (in this instance unnecessary)
# tn.write('{"appName": "Example", "appKey": "9f54141b4b4c567c558d3a76cb8d715cbde03096"}');
tn.write(('{"enableRawOutput": true, "format": "Json"}').encode('ascii'))

blink_or_not = raw_input('Non-zero blink(1) or zero blink(0): ')

outfile = "null";
if len(sys.argv) > 1:
    outfile = sys.argv[len(sys.argv) - 1];
    outfptr = open(outfile, 'w');

eSenseDict = {'attention': 0, 'meditation': 0};
waveDict = {'lowGamma': 0, 'highGamma': 0, 'highAlpha': 0, 'delta': 0, 'highBeta': 0, 'lowAlpha': 0, 'lowBeta': 0,
            'theta': 0};
signalLevel = 0;

time_list = []

person_name = raw_input('Enter the name of the person: ')

while time.clock() < 1:
    blinkStrength = 0;
    # print time.clock(), " is the time"

    # line = tn.read_until('\r');
    line = tn.read_all();
    if len(line) > 0:
        timediff = time.clock() - start;
        dict = json.loads(str(line));
        print(line)
        if "poorSignalLevel" in dict:
            signalLevel = dict['poorSignalLevel'];
        if "blinkStrength" in dict:
            blinkStrength = dict['blinkStrength'];
        if "eegPower" in dict:
            waveDict = dict['eegPower'];
            eSenseDict = dict['eSense'];
        outputstr = str(timediff) + ", " + str(signalLevel) + ", " + str(blinkStrength) + ", " + str(
            eSenseDict['attention']) + ", " + str(eSenseDict['meditation']) + ", " + str(
            waveDict['lowGamma']) + ", " + str(waveDict['highGamma']) + ", " + str(waveDict['highAlpha']) + ", " + str(
            waveDict['delta']) + ", " + str(waveDict['highBeta']) + ", " + str(waveDict['lowAlpha']) + ", " + str(
            waveDict['lowBeta']) + ", " + str(waveDict['theta']);
        if blinkStrength == 0 and eSenseDict['attention'] == 0 and eSenseDict['meditation'] == 0 and waveDict[
            'lowGamma'] == 0 and waveDict['highGamma'] == 0 and waveDict['highAlpha'] == 0 and waveDict[
            'lowAlpha'] == 0 and waveDict['lowBeta'] == 0 and waveDict['highBeta'] == 0 and waveDict['delta'] == 0 and \
                waveDict['theta'] == 0:
            continue
        time_array = np.append(time_array, [timediff]);
        blinkStrength_values = np.append(blinkStrength_values, [blinkStrength]);
        lowGamma_values = np.append(lowGamma_values, [waveDict['lowGamma']]);
        highGamma_values = np.append(highGamma_values, [waveDict['highGamma']]);
        highAlpha_values = np.append(highAlpha_values, [waveDict['highAlpha']]);
        delta_values = np.append(delta_values, [waveDict['delta']]);
        lowBeta_values = np.append(lowBeta_values, [waveDict['lowBeta']]);
        highBeta_values = np.append(highBeta_values, [waveDict['highBeta']]);
        theta_values = np.append(theta_values, [waveDict['theta']]);
        lowAlpha_values = np.append(lowAlpha_values, [waveDict['lowAlpha']]);
        attention_values = np.append(attention_values, [eSenseDict['attention']]);
        meditation_values = np.append(meditation_values, [eSenseDict['meditation']]);
        # print outputstr;
        if blink_or_not:
            if blinkStrength:
                time_list.append(timediff)
            # print time_list
        else:
            if blinkStrength == 0:
                time_list.append(timediff)
        if outfile != "null":
            outfptr.write(outputstr + "\n");

# Data Recorded for a single person
data_row = pd.DataFrame(
    {'Name': person_name, 'attention': [attention_values], 'meditation': [meditation_values], 'delta': [delta_values],
     'theta': [theta_values], 'lowAlpha': [lowAlpha_values], 'highAlpha': [highAlpha_values],
     'lowBeta': [lowBeta_values], 'highBeta': [highBeta_values],
     'lowGamma': [lowGamma_values], 'highGamma': [highGamma_values], 'blinkStrength': [blinkStrength_values],
     'time': [time_array]})

'''
fd = open('darren_data_eeg_pre.csv','a')
fd.write(str(blink_label)+','+str(person_name)+','+str([attention_values])+','+str([blinkStrength_values])+','+str([delta_values])+','+
str([highAlpha_values])+','+str([highBeta_values])+','+str([highGamma_values])+','+str([lowAlpha_values])+','+str([lowBeta_values])+','+str([lowGamma_values])+','+
str([meditation_values])+','+str([theta_values])+','+str([time_array])+','+'\n')
fd.close()

'''
# dataset_pre = pd.read_csv('darren_data_eeg_pre.csv')
min_time_list = []

# Reading the data stored till now
# dataset = pd.read_csv('darren_data_eeg_pre.csv')

# dataset = dataset.append(pd.Series([person_name, [attention_values], [blinkStrength_values], [delta_values]
#                                        , [highAlpha_values], [highBeta_values], [highGamma_values], [lowAlpha_values],
#                                     [lowBeta_values], [lowGamma_values], [meditation_values],
#                                     [theta_values]],
#                                    index=['Name', 'attention', 'blinkStrength', 'delta', 'highAlpha', 'highBeta',
#                                           'highGamma', 'lowAlpha', 'lowBeta', 'lowGamma', 'meditation', 'theta']),
#                          ignore_index=True)
#
# # Appending and storing the data in the same csv
# # dataset.append(data_row)
# dataset.to_csv('darren_data_eeg_pre.csv')

tn.close();
# outfptr.close();
