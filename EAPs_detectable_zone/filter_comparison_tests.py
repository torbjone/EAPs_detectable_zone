import os
import sys
from os.path import join
from glob import glob
import posixpath
import json
import numpy as np
import neuron
import LFPy
import matplotlib
matplotlib.use("AGG")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, LineCollection
import elephant
import pandas as pd
from plotting_convention import mark_subplots, simplify_axes
import scipy.signal as ss

dt = 2**-5


# high-pass filter settings
Fs = 1000 / dt
N = 1  # filter order
rp = 0.1  # ripple in passband (dB)
rs = 40.  # minimum attenuation required in the stop band (dB)
fc = 300.  # critical frequency (Hz)

# filter coefficients on 'sos' format
sos_ellip = ss.ellip(N=N, rp=rp, rs=rs, Wn=fc, btype='hp', fs=Fs, output='sos')

filt_dict_high_pass = {'highpass_freq': 300,
                       'lowpass_freq': None,
                       'order': 1,
                       'filter_function': 'filtfilt',
                       'fs': 1 / (dt / 1000),
                       'axis': -1
                       }

cell_name = 'hay'
imem_eap_folder = join("..", "imem_EAPs")

imem_ufilt = np.load(os.path.join(imem_eap_folder, "imem_ufilt_%s.npy" % cell_name))[0, :]
imem_filt = elephant.signal_processing.butter(imem_ufilt, **filt_dict_high_pass)
#imem_filt2 = ss.butter(imem_ufilt, **filt_dict_high_pass)
imem_filt3 = ss.sosfiltfilt(sos_ellip, imem_ufilt)


tvec = np.arange(len(imem_ufilt)) * dt

fig = plt.figure(figsize=[9, 9])
plt.plot(tvec, imem_ufilt - np.mean(imem_ufilt), 'k')
plt.plot(tvec, imem_filt - np.mean(imem_filt), 'r')
plt.plot(tvec, imem_filt3 - np.mean(imem_filt3), 'b')

plt.savefig("filter_test.png")