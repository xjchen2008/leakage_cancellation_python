# https://github.com/sgoadhouse/msox3000/
# Copyright (c) 2021, Stephen Goadhouse <sgoadhouse@virginia.edu>
# Modified by Xingjian Chen 2021
# -------------------------------------------------------------------------------
# Get data capture from Agilent/KeySight MSOX6004A scope and save it to a file
#

# pyvisa 1.6 (or higher) (http://pyvisa.sourceforge.net/)
# pyvisa-py 0.2 (https://pyvisa-py.readthedocs.io/en/latest/)
#
# NOTE: pyvisa-py replaces the need to install NI VISA libraries
# (which are crappily written and buggy!) Wohoo!
#
# -------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import coe_wavetable_4096 as coe
import functions
import setup
import dsp_filters
import dsp_filters_BPF
from scipy.signal import butter, lfilter, freqz
import random
import sys
from time import sleep
from scipy import signal



def readosc(itt=1, filename=''):
    # Set to the IP address of the oscilloscope
    agilent_msox_3034a = os.environ.get('MSOX3000_IP', 'TCPIP0::169.254.199.3::INSTR')

    import argparse
    parser = argparse.ArgumentParser(
        description='Get a screen capture from Agilent/KeySight MSO3034A scope and save it to a file')

    # fn_ext = ".png"
    pn = dir_path = os.path.dirname(os.path.realpath(__file__))  # os.environ['HOME'] + "/Downloads"
    fn = pn + "/" + filename  # args.ofile[0]

    import msox3000.MSOX3000 as MSOX3000  # this is import from the local file in msox3000 folder.

    ## Connect to the Power Supply with default wait time of 100ms
    scope = MSOX3000(agilent_msox_3034a)
    scope.open()

    # print(scope.idn())
    # print("Output file: %s" % fn)
    # scope.hardcopy(fn) # what is this for?
    #scope.waveform(fn, '1', itt, points=500000)  # use this one for long chirp
    scope.waveform(fn, '1', itt, points=4000) # try with 1.6 mu chirp with no avg in /acquire
    # scope.waveform(fn, '1', itt, points=400000) # try with 1.6 mu chirp
    # scope.waveform(fn+"_3", '3', itt)  # , points=5000)
    # scope.waveform(fn+"_4", '4', itt)  # , points=5000)

    # scope.waveform(fn + "_2.csv", '1', points=500000)
    # scope.waveform(fn+"_2.csv", '2')
    # scope.waveform(fn+"_3.csv", '3')
    # scope.waveform(fn+"_4.csv", '4')
    # print('Osc Done')
    scope.close()


def plot_fft(x, fs):
    # fs = 1
    N = len(x)
    freq = np.fft.fftfreq(N, d=1. / fs)
    X = 20 * np.log10(np.fft.fft(x))
    X = X - max(X)  # normalization.
    # plt.figure()
    # plt.plot(freq/1e6, X)
    signal_bw_pos_upper = 40
    signal_bw_pos_lower = 20
    signal_bw_neg_upper = -20
    signal_bw_neg_lower = -40
    plt.axvline(signal_bw_pos_upper, color='k', ls='-.')
    plt.axvline(signal_bw_pos_lower, color='k', ls='-.')
    plt.axvline(signal_bw_neg_upper, color='k', ls='-.')
    plt.axvline(signal_bw_neg_lower, color='k', ls='-.')
    plt.xlabel('frequency [MHz]')
    plt.ylabel('Normalized Amplitude [dB]')


def readcsv(filename=''):
    df = pd.read_csv(filename)
    # rx0 = df.iloc[::40,:] # ::40 is for 16us. Decimation: e.g. Decimate to 4000 points from 4000000 points; The sample rate is set on Osc
    rx0 = df
    rx = rx0['Voltage (V)'].values
    # print('len(rx)=', len(rx))
    # df.plot(x = 'Time (s)', y='Voltage (V)')
    # plt.figure()
    # rx0.plot( x='Time (s)', y='Voltage (V)')
    # plt.plot(rx)
    # plt.xlabel('Sample number')
    # plt.ylabel('Voltage (V)')

    ##############################
    # DDS
    ##############################
    Tr = 20e-6  # 20us generated chirp
    N = len(rx)
    fs = N / Tr
    # print('fs=',fs/1e6, 'MHz')
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

    # plot_fft(rx, fs)
    # x_tx = coe.y_cx.real
    # plot_fft(x_tx, fs)
    # plot_fft(rx_all, fs)
    ##############################
    # plt.show()
    return rx


def distance2freq(distance):
    # https://matplotlib.org/stable/gallery/subplots_axes_and_figures/secondary_axis.html
    freq = distance / (c) * coe.k * 2.0
    return freq / 1e6  # MHz


def freq2distance(freq):
    distance = c * freq * 1e6 / coe.k / 2.0  # x * 180 / np.pi
    return distance


def data_process(rx, rx_cal):
    N = len(rx)
    rx_cpx = signal.hilbert(rx)
    rx_cpx_cal = signal.hilbert(rx_cal)
    x = coe.y_cx

    RX = np.fft.fft(rx)
    RX_cal = np.fft.fft(rx_cal)
    RX_cpx = np.fft.fft(rx_cpx)
    RX_cpx_cal = np.fft.fft(rx_cpx_cal)
    X = np.fft.fft(x)
    X_real = np.fft.fft(x.real)

    amp_diff = rx - x.real
    amp_diff_cal = rx_cal - x.real

    phase_rx_cal = np.unwrap(np.angle(rx_cpx_cal))  # must in unit of radian
    phase_rx = np.unwrap(np.angle(rx_cpx))  # must in unit of radian
    phase_x = np.unwrap(np.angle(x))  # must in unit of radian

    phase_diff = phase_rx - phase_x
    phase_diff_cal = phase_rx_cal - phase_x
    fs = 250e6
    freq = np.fft.fftfreq(N, d=1. / fs)
    dt = 1 / fs
    derivative_phase_x = np.diff(np.angle(
        x)) / dt  # make derivative of phase to check frequency. https://stackoverflow.com/questions/16841729/how-do-i-compute-the-derivative-of-an-array-in-python/19459160
    n = np.linspace(0, N, N)
    plt.plot(n, rx_cpx.real, n, rx_cpx.imag)
    plt.title('Hilber transform of the received chirp( cos(x)-> exp[jx]')
    ###########
    plt.figure()
    plt.plot(n, x.real, n, x.imag)
    plt.title('An Ideal Chirp')
    plt.xlabel('Sample Number')
    plt.ylabel('Amplitude [dB]')
    plt.legend(['Real Part', 'Imaginary Part'])
    plt.grid()
    # ########## The phase of the chirp is parabolic since the derivative of up-parabolic function is linear increase
    # function that is the freq.
    plt.figure()
    plt.plot(n, phase_x, n, np.angle(x))
    plt.title('Phase of a Chirp')  # comparision of wrapped and unwrapped phase')
    plt.xlabel('Sample Number')
    plt.ylabel('Phase [radian]')
    plt.legend(['Unwrapped Phase', 'Wrapped Phase'])
    plt.grid()
    ###########
    plt.figure()
    plt.plot(n, phase_x, n, phase_rx)
    plt.title('time domain comparision of phase of ideal chirp and received chirp')
    ###########
    plt.figure()
    plt.plot(n, phase_diff, n, phase_diff_cal)
    plt.title('Transmission Line: Phase Error Correction')  # between ideal chirp and received chirp')
    plt.xlabel('Samples')
    plt.ylabel('Phase [radian]')
    plt.legend(['Without EQ-DPD', 'With EQ-DPD'])
    plt.grid()
    ###########
    plt.figure()
    plt.plot(freq / 1e6, 20 * np.log10(RX / max(abs(RX))), freq / 1e6, 20 * np.log10(RX_cal / max(abs(RX_cal)))
             )  # ,freq/1e6, 20*np.log10(X_real/max(abs(X_real))))
    plt.title('Transmission Line: Amplitude Response Correction')
    plt.xlabel('Frequency [MHz]')
    plt.ylabel('Normalized Amplitude [dB]')
    plt.legend(['Without EQ-DPD', 'With EQ-DPD'])
    plt.grid()
    '''
    ###########
    plt.figure()
    plt.plot(n[0:N-1], derivative_phase_x)
    plt.title('Check the frequency change of the chirp')
    ###########
    plt.figure()
    plt.plot(freq/1e6, np.unwrap(np.angle(RX_cpx)), freq/1e6, np.unwrap(np.angle(X)))
    plt.xlabel('freq [MHz]')
    plt.title('The frequency domain comparision for unwrapped phase between the ideal chirp and received signal')
    ###########
    plt.figure()
    plt.plot(freq/1e6,20*np.log10(abs(RX)), freq/1e6, 20*np.log10(abs(X_real)))
    plt.xlabel('freq [MHz]')
    plt.title('Amplitude comparision between the received signal and ideal chirp')
    '''


if __name__ == '__main__':
    c = 3e8
    j = 1j
    fs = coe.fs  # Sampling freq
    N = coe.N  # This also limit the bandwidth. And this is determined by fpga LUT size.
    T = N / fs  # T=N/fs#Chirp Duration
    t = np.linspace(0, T, N)
    f0 = coe.f0  # Start Freq
    f1 = coe.f1  # fs/2=1/2*N/T#End freq
    K = (f1 - f0) / T  # chirp rate = BW/Druation
    f = np.linspace(0, fs - 1, N)
    freq = np.fft.fftfreq(N, d=1 / fs)
    distance = c * freq / K / 2.0
    win = np.blackman(N)
    # readosc(filename='data/BPF_antenna_4000_indoor2.csv')
    # readosc(filename='data/HighRes_40000_indoor.csv')
    # readosc(filename='data/test_sine.csv')
    # readosc(filename='data/test_blackman.csv')
    # readosc(filename='data/output_cal_loopback_antenna_128_20MHz_5mV_2BBAmp_gen10dBm.csv')  # antenna measurement
    # readosc(filename='output_cal_loopback_antenna_128_20MHz_5mV_2BBAmp.csv')  # antenna measurement
    # readosc(filename='output_cal_loopback_TL_IQ_128_20MHz_500mV.csv')  # trasnmissionline measurement
    # readosc(filename='output_cal_loopback_TL_ch1.csv')  # TL measurement
    # readosc(filename='output_cal_loopback_TL_ch1_1MHz.csv')  # TL measurement
    # readosc(filename='output_cal_loopback_NoAntenna_origianlx.csv')  # antenna measurement
    # readosc(filename='output_cal_antenna_ch2_origianlx.csv')  # antenna measurement
    # readosc(filename='output_cal_antenna_ch2_origianlx_field.csv')  # antenna measurement
    # readosc(filename='output_cal_antenna_ch2_origianlx_field_measure.csv')  # antenna measurement
    # readosc(filename='output_cal_antenna_ch2_origianlx_field_measure2.csv')  # antenna measurement
    # readosc(filename='output_cal_antenna_ch2_origianlx_field_measure3.csv')  # antenna measurement
    # readosc(filename='output_cal_antenna_ch2_origianlx_field_measure3_outdoor1.csv')  # antenna measurement
    # readosc(filename='original_indoor65536_1.csv')  # antenna measurement
    # readosc(filename='original_indoor_noAvg_1.csv')  # antenna measurement
    # readosc(filename='original_indoor_N400000_1.csv')  # antenna measurement

    # readosc(filename='output_cal_antenna_ch2.csv')  # antenna measurement
    # readosc(filename='output_cal_antenna_ch2_EQ.csv')
    # readosc(filename='output_cal_antenna_ch2_EQ_ch1actualCopy.csv')
    # readosc(filename='output_cal_actual_ch1_copy_ch2.csv')
    # readosc(filename='output_cal_Mixer_ch2.csv')
    # readosc(filename='output_cal_Mixer_ch2_after_EQ.csv')
    # readosc(filename='output_cal_1.csv')
    # readosc(filename='output_baseband_idealchirp_1MHz.csv')

    rx = np.zeros(N)
    tx = np.zeros(N)

    ###################
    # Taking Osc measurement
    ###################
    avg = False
    if avg == True:
        #If do avg measurement for generate template tx signal
        #for itt in range(1000):
            # Save multiple measurements for calculating the averaged chirp.
            # readosc(filename='data/avg/BPF_antenna_500000_outdoor_40_60MHz_chirp_Noavg_measure_cancellation1'+str(itt)+'.csv')
            #readosc(filename='data/avg/TL_500000_indoor_40_60MHz_chirp_Noavg_measure' + str(itt) + '.csv')
        #    readosc(filename='data/avg/antenna_3999_indoor_40_60MHz_chirp_Noavg_measure_afterCanc2_D100_delaym40' + str(itt) + '.csv')

        for itt in range(1000):
            # Read multiple measurements for calculating the averaged chirp.
            rx += readcsv(filename='data/avg/antenna_3999_indoor_40_60MHz_chirp_Noavg_measure_afterCanc2_D100_delaym40' + str(itt) + '.csv')
        #np.save(file=setup.file_tx, arr=rx)  # Comment out if not making template. Store the chirp
    else:
        # if not do avg measurement use the following code:
        readosc(filename=setup.file_rx)
        rx = readcsv(filename=setup.file_rx)
        # rx = np.load(file='data/avg/BPF_antenna_500000_outdoor_40_60MHz_chirp_100avg_measure_cancellation1.npy') # readcsv(filename='data/avg/BPF_antenna_500000_outdoor_40_60MHz_chirp_Noavg_measure_cancellation1.csv')
    #################################
    #
    #################################
    #rx = dsp_filters_BPF.run(rx)
    rx = rx / max(rx)  # normalization
    rx_cx = signal.hilbert(rx)
    # rx_cx = dsp_filters.main(signal=rx_cx, order=6, fs=fs, cutoff=40e6, duration=T)
    RX_cx = np.fft.fft(rx_cx)
    RX_cx[-1] = 0
    RX_cx[0:100] = 0
    RX_cx[-1 - 100:] = 0
    rx_cx = np.fft.ifft(RX_cx)
    tx = np.load(
        file=setup.file_tx)  # readcsv(filename='data/avg/BPF_TL_500000_indoor_40_60MHz_chirp_Noavg.csv')
    tx_cx = signal.hilbert(tx)
    plt.plot(np.fft.fftshift(freq) / 1e6, np.fft.fftshift(functions.normalize(20 * np.log10(abs(np.fft.fft(rx_cx.real))))))
    plt.title('Normalized Received Signal in Frequency Domain')
    plt.xlabel('Frequency [MHz]')
    plt.ylim([-60,10])
    plt.grid(True)
    # plt.plot(freq, 20 * np.log10(abs(np.fft.fft(coe.y_cx))))
    # plt.plot(freq/1e6, 20 * np.log10(abs(np.fft.fft(np.multiply(coe.y_cx.real,win)))))
    # plt.plot(freq, 20 * np.log10(abs(np.fft.fft(signal.hilbert(coe.y_cx.real)))))
    # plt.plot(freq, 20 * np.log10(abs(np.fft.fft(coe.y_cx))))
    # plt.show()

    # pc = functions.PulseCompr(rx=signal.hilbert(coe.y_cx.real), tx=signal.hilbert(coe.y_cx.real), win=win)
    # pc = functions.PulseCompr(rx=coe.y_cx, tx=coe.y_cx, win=win)
    # pc = functions.PulseCompr(rx=coe.y_cx.real,tx =coe.y_cx.real, win=win)
    # pc = functions.PulseCompr(rx=np.concatenate([rx,rx]), tx=np.concatenate([rx,rx]), win= win)
    # rx_cx = np.multiply(rx_cx, win)
    pc = functions.normalize(functions.PulseCompr(rx=rx_cx, tx=tx_cx, win=win))
    #pc = functions.PulseCompr(rx=rx_cx, tx=tx_cx, win=win)
    # pc = functions.PulseCompr(rx=rx_cx, tx=rx_cx, win=win)

    # plt.plot(t, np.fft.ifft(pc).real, t, np.fft.ifft(pc).imag)
    # plt.show()
    # rx_cx_upsamp = functions.upsampling(rx_cx, 1)
    # pc = functions.PulseCompr(rx=rx_cx_upsamp, tx=rx_cx_upsamp, win=win)
    # pc = functions.downsampling(pc, 1)

    pc_timedomain = np.fft.ifft(pc)
    pc_timedomain_LPF = dsp_filters.main(signal=pc_timedomain, order=20, fs=fs, cutoff=10e6, duration=T)
    pc_timedomain_LPF_win = np.multiply(pc_timedomain_LPF, np.blackman(N))  # .reshape([N,1]))
    pc_freqdomain_LPF = np.fft.fft(pc_timedomain_LPF_win)
    # pc_log = 20 * np.log10(abs(pc_freqdomain_LPF)) # with LPF for sretch method
    # pc_log = 20 * np.log10(abs(pc))# no LPF
    # pc_log = pc_log - max(pc_log)  # normalization
    fig, ax = plt.subplots()
    ax.plot(np.fft.fftshift(distance), np.fft.fftshift(pc), '*-')
    ax.set_xlim([-200, 500])
    ax.set_ylim([-90, 20])
    plt.xlabel('Distance [m]')
    secax = ax.secondary_xaxis('top', functions=(distance2freq, freq2distance))
    secax.set_xlabel('Frequency [MHz]')
    plt.grid()
    plt.ylabel('Amplitude [dB]')
    plt.title('Pulse Compression of Antenna Loopback')
    # plt.axvline(30, color='k', ls='-.')
    # plt.axvline(20, color='b', ls='-.')
    # plt.axvline(40, color='b', ls='-.')
    # plt.axvline(50, color='r', ls='-.')
    # plt.axvline(70, color='r', ls='-.')
    # plt.axvline(60, color='k', ls='-.')
    # plt.axvline(90, color='k', ls='-.')
    print(len(rx))
    # readosc(filename='output_cal_2.csv')
    # rx_cal = tx  # readcsv(filename='output_cal_2.csv') # the calibrated chirp.
    # data_process(rx, rx_cal)
    # plt.plot(rx)
    plt.show()
