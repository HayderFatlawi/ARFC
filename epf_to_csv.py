from pyedflib import highlevel
import numpy as np
import pandas as pd

signals, signal_headers, header = highlevel.read_edf('path_to_edf_file')
print(signal_headers[0]['sample_rate']) # prints 256

#set sample rate
fs = 256  

fft_vals = np.absolute(np.fft.fft(signals))
fft_vals2= np.ndarray.transpose(fft_vals)
R=fft_vals2.reshape((3600, 256, 23)).sum(axis=1)
fft_freq = np.fft.rfftfreq(len(signals), 1.0/fs)

np.savetxt("path_to_csv_file", R, delimiter=',')