import json
import subprocess
from telnetlib import Telnet

import numpy as np
from sklearn.externals import joblib


class Classify:
    def __init__(self):
        self.tn = Telnet('localhost', 13854)
        self.tn.write('{"enableRawOutput": true, "format": "Json"}'.encode('ascii') + b"\n")
        self.wave_dict = {'lowGamma': 0, 'highGamma': 0, 'highAlpha': 0,
                          'delta': 0, 'highBeta': 0, 'lowAlpha': 0,
                          'lowBeta': 0, 'theta': 0}
        self.signal_level = 0
        self.raw_eeg = 0
        # load the model from disk
        self.filename = '../models/rfc_2_8vals_study_binary_model.pkl'
        self.loaded_model = joblib.load(self.filename)

    def prediction(self, array):
        return self.loaded_model.predict(array)

    def telnet_conn(self):
        return self.tn

    def check_zero_vals(self):
        return self.wave_dict['lowGamma'] == 0 and self.wave_dict['highGamma'] == 0 and \
               self.wave_dict['highAlpha'] == 0 and self.wave_dict['lowAlpha'] == 0 and \
               self.wave_dict['lowBeta'] == 0 and self.wave_dict['highBeta'] == 0 and \
               self.wave_dict['delta'] == 0 and self.wave_dict['theta'] == 0

    def make_output_str(self):
        return str(self.raw_eeg) + str(self.wave_dict['delta']) + ", " + str(self.wave_dict['theta']) + ", " + \
               str(self.wave_dict['lowAlpha']) + ", " + str(self.wave_dict['highAlpha']) + ", " \
               + str(self.wave_dict['lowBeta']) + ", " + str(
            self.wave_dict['highBeta']) + ", " + str(
            self.wave_dict['lowGamma']) + ", " + str(self.wave_dict['highGamma'])

    def out_to_float_array(self):
        eeg = float(str(self.raw_eeg))
        d = float(str(self.wave_dict['delta']))
        t = float(str(self.wave_dict['theta']))
        la = float(str(self.wave_dict['lowAlpha']))
        ha = float(str(self.wave_dict['highAlpha']))
        lb = float(str(self.wave_dict['lowBeta']))
        hb = float(str(self.wave_dict['highBeta']))
        lg = float(str(self.wave_dict['lowGamma']))
        hg = float(str(self.wave_dict['highGamma']))
        return [[eeg, d, t, la, ha, lb, hb, lg, hg]]

    @staticmethod
    def normalise_values(d_array):
        # standardisation works best for Random Forest
        d_array = np.asarray(d_array)
        d_array = d_array.astype('float32')
        d_array = d_array.astype('float32')
        mean = d_array.mean()
        d_array -= mean
        std = d_array.std()
        d_array /= std
        return d_array

    def set_wave_dict(self, wave_dict):
        self.wave_dict = wave_dict

    def set_raw_eeg(self, raw_eeg):
        self.raw_eeg = raw_eeg

    def process_values(self):
        # outputstr = self.make_output_str()
        data_array = self.out_to_float_array()
        data_array = Classify.normalise_values(data_array)
        print(data_array)
        result = self.prediction(data_array)
        print(result)
        if result == [1]:
            subprocess.call(["afplay", "dit.wav"])

    def read_tn(self):
        while True:
            line = self.tn.read_until(b'\r')
            line = line.decode()
            if len(line) > 0:
                dict = json.loads(str(line))
                if "rawEeg" in dict:
                    self.set_rawEeg(dict['rawEeg'])
                if "eegPower" in dict:
                    self.set_waveDict(dict['eegPower'])
                if self.check_zero_vals():
                    continue
                self.process_values()

    def close_tn(self):
        self.tn.close()

#
# df1 = pd.DataFrame(data_array,
#                            columns=['eegRaw', 'delta', 'theta', 'alphaLow', 'alphaHigh',
#                                     'betaLow', 'betaHigh', 'gammaLow', 'gammaMid'])
# df2.append(df1)
