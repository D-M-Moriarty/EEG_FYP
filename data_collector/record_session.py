import subprocess
import time

import cv2

from NeuroPy.NeuroPy.NeuroPy import NeuroPy

neuroPy = NeuroPy('/dev/cu.MindWaveMobile-SerialPo-3')
neuroPy.start()
image_path = "/Users/darrenmoriarty/ml/EEG_FYP/image_processing/images/arrow_down0.4.png"
output_file = '../../data_files/serial_connection_on_mac_recordings/move_drone_forward_10_5mins.csv'
# output_file = '../../data_files/serial_connection_on_mac_recordings/rest_17ishmins.csv'
out = open(output_file, 'w')
PERIOD_OF_TIME = 60 * 5  # 5min
header = 'eegRawValue,delta,theta,alphaLow,alphaHigh,betaLow,betaHigh,gammaLow,gammaMid,action'
out.write(header + '\n')


def get_usable_waves():
    return str(neuroPy.rawValue) + ',' + str(neuroPy.delta) + ',' + str(neuroPy.theta) + ',' + \
           str(neuroPy.lowAlpha) + ',' + str(neuroPy.highAlpha) + ',' + str(neuroPy.lowBeta) + ',' + \
           str(neuroPy.highBeta) + ',' + str(neuroPy.lowGamma) + ',' + str(neuroPy.midGamma)


def read_waves(action):
    while True:
        print(get_usable_waves() + ',' + action)
        out.write(get_usable_waves() + ',' + action + '\n')
        time.sleep(0.1)  # Don't eat the CPU cycles
        if time.time() > start + PERIOD_OF_TIME:
            break
    cv2.destroyAllWindows()

try:
    time.sleep(3)
    # action = raw_input("enter the action being performed: ")
    start = time.time()
    read_waves('rest')
    for i in range(5):
        subprocess.call(["afplay", "/Users/darrenmoriarty/ml/EEG_FYP/sounds/dit.wav"])
    neuroPy.stop()

except:
    print("exit")
    neuroPy.stop()
