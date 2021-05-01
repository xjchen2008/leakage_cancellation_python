import numpy as np
import timeit
import os
import struct
from numpy import fft
import skrf as rf
import time
import matplotlib.pyplot as plt
import coe_wavetable_4096 as coe
import functions
from functions import plot_freq_db as plot_freq
from functions import PulseCompr, normalize
from numpy.fft import fftshift

freq = (coe.freq/1e6)
distance = (coe.distance)
N = coe.N
mean = 0
std = 0.01
noise1 = np.random.normal(mean, std, size=N)
delay = 1000
win = np.blackman(N)
win1 = np.hamming(N)
win2 = np.hanning(N)
x = coe.y_cx + noise1
x_delay = coe.y_cx_0_delay
x_win = np.multiply(x, win)
x_win1 = np.multiply(x, win1)
x_win2 = np.multiply(x, win2)
x_win_delay = np.roll(x_win, -delay)

plot_freq(freq, x, 'b')
plot_freq(freq, x_win,'k')
plot_freq(freq, x_win1,'m')
plot_freq(freq, x_win2,'g')
plt.title('Frequency Response of a Linear Chirp')
plt.legend(['No window', 'Blackman window', 'Hamming window', 'Hanning window'])
plt.xlabel('Frequency [MHz]')
plt.ylabel('Amplitude [dB]')
# pulse compression
pc = PulseCompr(rx=x_delay, tx = x, win = win1)
pc_win = PulseCompr(rx=x_win_delay,tx= x_win, win =win1)
pc_win1 = PulseCompr(rx=x_win_delay,tx= x_win1, win =win1)
pc_win2 = PulseCompr(rx=x_win_delay, tx=x_win2, win =win1)

plt.figure()
plt.plot(fftshift(freq), fftshift(pc), 'b')
plt.plot(fftshift(freq), fftshift(pc_win),'k')
plt.plot(fftshift(freq), fftshift(pc_win1),'m')
plt.plot(fftshift(freq), fftshift(pc_win2),'g')
plt.xlabel('Frequency [MHz]')
plt.ylabel('Amplitude [dB]')
plt.grid()
plt.title('Pulse Compression')
plt.legend(['No window', 'Blackman window', 'Hamming window', 'Hanning window'])

plt.show()