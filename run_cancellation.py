import functions
import numpy as np
import matplotlib.pyplot as plt
import readosc
import coe_wavetable_4096 as coe
import functions
import dsp_filters
from scipy import signal

tx = readosc.readcsv(filename='output_cal_loopback_NoAntenna_origianlx.csv')
rx = readosc.readcsv(filename='output_cal_antenna_ch2_origianlx_field.csv')
rx = rx/max(rx) # normalization
rx_cx = signal.hilbert(rx)


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


# For test, this skips the orthognalization!
w, H = functions.channel_est(psi_orth = tx, y_cx_received = rx)

x_canc = np.squeeze(np.dot(H,w)) # np.squeeze is make shape of x_canc same for later calculation
plt.plot(x_canc)
plt.title('The cancellation signal')
y_canc = rx - x_canc.T
plt.figure()
plt.plot(y_canc)
plt.title('Remaining received signal after cancellation ')
# Pulse compression after cancellation
PC = functions.PulseCompr(rx=y_canc,tx=tx,win=np.blackman(N))
plt.figure()
plt.plot(distance,PC,'k',label='After Leakage Cancellation')
#plt.xlim((-10,800))
plt.xlabel('Distance in meter')
plt.ylabel('Power in dB')
plt.grid()
# Pulse compression before cancellation
PC = functions.PulseCompr(rx = rx,tx = tx,win = np.blackman(N))
plt.plot(distance,PC,'b',label='Before Leakage Cancellation')
plt.title('Pulse Compression')
plt.legend()

# Pulse compression with just cancellation signal
PC = functions.PulseCompr(rx = x_canc.T,tx=tx,win=np.blackman(N))
plt.figure()
plt.plot(distance,PC,'r')
#plt.xlim((-10,800))
plt.xlabel('Distance in meter')
plt.ylabel('Power in dB')
plt.title('Pulse compression with Cancellation Signal')
plt.grid()
plt.show()