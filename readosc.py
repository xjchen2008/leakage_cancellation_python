#https://github.com/sgoadhouse/msox3000/
# Copyright (c) 2021, Stephen Goadhouse <sgoadhouse@virginia.edu>
# Modified by Xingjian Chen 2021
#-------------------------------------------------------------------------------
# Get data capture from Agilent/KeySight MSOX6004A scope and save it to a file
#

# pyvisa 1.6 (or higher) (http://pyvisa.sourceforge.net/)
# pyvisa-py 0.2 (https://pyvisa-py.readthedocs.io/en/latest/)
#
# NOTE: pyvisa-py replaces the need to install NI VISA libraries
# (which are crappily written and buggy!) Wohoo!
#
#-------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import coe_wavetable_4096 as coe
import dsp_filters
from scipy.signal import butter, lfilter, freqz
import random
import sys
from time import sleep


def readosc(itt = 0, filename=''):
    # Set to the IP address of the oscilloscope
    agilent_msox_3034a = os.environ.get('MSOX3000_IP', 'TCPIP0::169.254.199.3::INSTR')

    import argparse
    parser = argparse.ArgumentParser(
        description='Get a screen capture from Agilent/KeySight MSO3034A scope and save it to a file')

    # fn_ext = ".png"
    pn = dir_path = os.path.dirname(os.path.realpath(__file__))  # os.environ['HOME'] + "/Downloads"
    fn = pn + "/" +filename  # args.ofile[0]

    import msox3000.MSOX3000 as MSOX3000 # this is import from the local file in msox3000 folder.

    ## Connect to the Power Supply with default wait time of 100ms
    scope = MSOX3000(agilent_msox_3034a)
    scope.open()

    #print(scope.idn())
    #print("Output file: %s" % fn)
    #scope.hardcopy(fn) # what is this for?
    scope.waveform(fn, '1', itt, points=5000)
    # scope.waveform(fn+"_2.csv", '2')
    # scope.waveform(fn+"_3.csv", '3')
    #scope.waveform(fn+"_4.csv", '4')
    #print('Osc Done')
    scope.close()


def plot_fft(x, fs):
    # fs = 1
    N = len(x)
    freq = np.fft.fftfreq(N, d=1. / fs)
    X = 20*np.log10(np.fft.fft(x))
    X = X-max(X) # normalization.
    #plt.figure()
    #plt.plot(freq/1e6, X)
    signal_bw_pos_upper = 40
    signal_bw_pos_lower = 20
    signal_bw_neg_upper = -20
    signal_bw_neg_lower = -40
    plt.axvline(signal_bw_pos_upper, color='k',ls='-.')
    plt.axvline(signal_bw_pos_lower, color='k',ls='-.')
    plt.axvline(signal_bw_neg_upper, color='k',ls='-.')
    plt.axvline(signal_bw_neg_lower, color='k',ls='-.')
    plt.xlabel('frequency MHz')
    plt.ylabel('Normalized Amplitude dB')


def readcsv(filename=''):
    df = pd.read_csv(filename)
    #rx0 = df.iloc[::40,:] # ::40 is for 16us. Decimation: e.g. Decimate to 4000 points from 4000000 points; The sample rate is set on Osc
    rx0 = df
    rx = rx0['Voltage (V)'].values
    #print('len(rx)=', len(rx))
    #df.plot(x = 'Time (s)', y='Voltage (V)')
    #plt.figure()
    #rx0.plot( x='Time (s)', y='Voltage (V)')
    #plt.plot(rx)
    #plt.xlabel('Sample number')
    #plt.ylabel('Voltage (V)')


    ##############################
    # DDS
    ##############################
    Tr = 20e-6  # 20us generated chirp
    N = len(rx)
    fs = N / Tr
    #print('fs=',fs/1e6, 'MHz')
    t = np.linspace(0, Tr, N)
    rx_convert = rx * np.cos(2 * np.pi * 1.4135e9 * t)
    '''
    ############################
    # low-pass filtering
    ############################
    # Filter requirements.
    order = 6
    fs = 2e9  # sample rate, Hz
    cutoff = 400e6  # desired cutoff frequency of the filter, Hz

    # Get the filter coefficients so we can check its frequency response.
    b, a = dsp_filters.butter_lowpass(cutoff, fs, order)

    # Plot the frequency response.
    w, h = dsp_filters.freqz(b, a, worN=8000)
    plt.subplot(2, 1, 1)
    plt.plot(0.5 * fs * w / np.pi, np.abs(h), 'b')
    plt.plot(cutoff, 0.5 * np.sqrt(2), 'ko')
    plt.axvline(cutoff, color='k')
    plt.xlim(0, 0.5 * fs)
    plt.title("Lowpass Filter Frequency Response")
    plt.xlabel('Frequency [Hz]')
    plt.grid()

    # Demonstrate the use of the filter.
    # First make some data to be filtered.
    T = 2e-6  # seconds
    n = int(T * fs)  # total number of samples
    t = np.linspace(0, T, n, endpoint=False)
    # "Noisy" data.  We want to recover the 1.2 Hz signal from this.
    #data = np.sin(1.4135e9 * 2 * np.pi * t) + 1.5 * np.cos(20e6 * 2 * np.pi * t) + 0.5 * np.sin(100e6 * 2 * np.pi * t)
    data = rx
    # Filter the data, and plot both the original and filtered signals.
    y = dsp_filters.butter_lowpass_filter(data, cutoff, fs, order)

    plt.subplot(2, 1, 2)
    plt.plot(t / 1e-6, data, 'b-', label='data')
    plt.plot(t / 1e-6, y, 'g-', linewidth=2, label='filtered data')
    plt.xlabel('Time [usec]')
    plt.grid()
    plt.legend()

    plt.subplots_adjust(hspace=0.35)
    '''
    ###################
    #
    ###################

    #plot_fft(rx, fs)
    #x_tx = coe.y_cx.real
    #plot_fft(x_tx, fs)
    #plot_fft(rx_all, fs)
    ##############################
    #plt.show()
    return rx


if __name__ == '__main__':
    readosc(filename='output_cal_1.csv')
    rx = readcsv(filename='output_cal_1.csv')
    print(rx)