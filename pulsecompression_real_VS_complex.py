import numpy as np
import matplotlib.pyplot as plt
import coe_wavetable_4096 as coe
from scipy import signal
import functions
from readosc import readcsv


t = coe.t
freq = coe.freq
y_cx = coe.y_cx  # original complex signal
y_r = y_cx.real  # real part of the original complex signal
y_i = y_cx.imag  # imaginary part of the original complex signal
y_hb = signal.hilbert(y_r)  # Hilbert transform of the real part of the original complex signal
y_hb_r = y_hb.real  # real part of the Hilbert transform signal
y_hb_i = y_hb.imag  # imaginary part of the Hilbert transform signal
#rx = readcsv(filename='output_cal_loopback_TL_IQ_128_20MHz_500mV.csv_4')
rx = readcsv(filename='data/test.csv')
#rx = readcsv(filename='output_cal_loopback_antenna_128_20MHz_5mV_2BBAmp.csv')
#rx = readcsv(filename='output_cal_loopback_antenna_128_20MHz_5mV.csv')
rx =  rx/max(rx)
rx_hb = signal.hilbert(rx)
rx_hb_r = rx_hb.real
rx_hb_i = rx_hb.imag

rx_I = readcsv(filename='data/output_cal_loopback_TL_IQ_128_20MHz_500mV.csv_4')
rx_Q = readcsv(filename='data/output_cal_loopback_TL_IQ_128_20MHz_500mV.csv_3')
rx_IQ_cx = rx_I + 1j * rx_Q


RX_HB = 20*np.log10(np.fft.fft(rx_hb))
RX_HB = (np.fft.fft(rx_hb))

N = len(y_cx)
win = np.blackman(N)
sig = np.multiply(np.multiply(rx, win), 1) # need this third window
SIG = np.fft.fft(sig)
SIG_log = 20*np.log10(SIG)
#plt.plot(freq, SIG, freq, RX_HB)



pc = functions.PulseCompr(sig, sig, win)
pc_log = 20* np.log10(abs(pc))
pc_log_normal = pc_log - np.max(pc_log)
plt.figure()
plt.plot(freq, pc_log_normal)
plt.title('Pulse Compression for the Signal rx_hb')
#plt.figure()
#plt.plot(pc)

pc1 = np.convolve(SIG,np.conj(SIG))
pc1_log = 20*np.log10(abs(pc1))
'''plt.figure()
plt.plot(SIG.imag)
plt.title('Imaginary part of SIG in frequency domain')
plt.figure()
plt.plot(SIG.real)
plt.title('Real part of SIG in frequency domain')'''
plt.figure()
plt.plot(pc1_log)
plt.title('Linear Convolution')

'''
error_r = y_r - y_hb_r
error_i = y_i - y_hb_i
print(np.max(error_i))
plt.figure()
plt.plot(t,error_i)
'''
plt.show()