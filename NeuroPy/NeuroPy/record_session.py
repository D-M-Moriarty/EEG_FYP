import time

from click._compat import raw_input

from NeuroPy import NeuroPy

neuroPy = NeuroPy()
neuroPy.start()
output_file = '../../pikachu.csv'
out = open(output_file, 'w')
PERIOD_OF_TIME = 300  # 5min
header = 'eegRawValue,delta,theta,alphaLow,alphaHigh,betaLow,betaHigh,gammaLow,gammaMid,action'
out.write(header + '\n')


def get_usable_waves():
    return str(neuroPy.rawValue) + ', ' + str(neuroPy.delta) + ', ' + str(neuroPy.theta) + ', ' + \
           str(neuroPy.lowAlpha) + ', ' + str(neuroPy.highAlpha) + ', ' + str(neuroPy.lowBeta) + ', ' + \
           str(neuroPy.highBeta) + ', ' + str(neuroPy.lowGamma) + ', ' + str(neuroPy.midGamma)


try:
    time.sleep(3)
    action = raw_input("enter the action being performed: ")
    start = time.time()
    while True:
        print(get_usable_waves())
        out.write(get_usable_waves() + ',' + action + '\n')
        time.sleep(0.1)  # Don't eat the CPU cycles
        if time.time() > start + PERIOD_OF_TIME:
            break

except:
    print("exit")
    neuroPy.stop()
