import coe_wavetable_4096 as coe
import numpy as np
import matplotlib.pyplot as plt
import setup
import functions
from numpy.fft import fftshift

N = coe.N
x = coe.y_cx
x_LO= coe.y_cx_sine2
x_RF= x*x_LO
freq = coe.freq
functions.plot_freq_db(freq, x_RF)
win = np.blackman(N)
# Pulse Compression
pc_LO = functions.PulseCompr(rx=x_LO, tx=x_LO, win=win, unit='log')
pc_RF = functions.PulseCompr(rx=x_RF, tx=x_RF, win=win, unit='log')
plt.figure()
plt.plot(fftshift(freq), fftshift(pc_LO), fftshift(freq), fftshift(pc_RF))
plt.grid()
plt.title('Pulse Compression')
plt.xlabel('Frequency [MHz]')
plt.ylabel('Amplitude [dB]')
#functions.plot_freq_db(freq, pc_LO)
plt.show()
