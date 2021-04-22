import functions
import numpy as np
import matplotlib.pyplot as plt
import readosc
import coe_wavetable_4096 as coe
import functions
import dsp_filters
from scipy import signal


def LPF_beatfreq(pc):
    pc_timedomain = np.fft.ifft(pc)
    pc_timedomain_LPF = dsp_filters.main(signal=pc_timedomain, order=6, fs=fs, cutoff=50e6, duration=T)
    pc_timedomain_LPF_win = np.multiply(pc_timedomain_LPF, np.blackman(N))  # .reshape([N,1]))
    pc_freqdomain_LPF = np.fft.fft(pc_timedomain_LPF_win)
    # pc_log = 20 * np.log10(abs(pc_freqdomain_LPF))  # with LPF for sretch method
    # pc_log = 20 * np.log10(abs(pc))# no LPF
    #pc_log = pc_log - max(pc_log)  # normalization
    return pc_freqdomain_LPF


tx = readosc.readcsv(filename='output_cal_loopback_NoAntenna_origianlx.csv')
rx_meas = readosc.readcsv(filename='output_cal_antenna_ch2_origianlx_field_measure.csv')
rx_meas2 = readosc.readcsv(filename='output_cal_antenna_ch2_origianlx_field_measure2.csv')
rx_meas3 = readosc.readcsv(filename='output_cal_antenna_ch2_origianlx_field_measure3.csv')
#rx_meas3 = readosc.readcsv(filename='output_cal_antenna_ch2_origianlx_field_measure3_outdoor1.csv')

rx_meas = rx_meas/max(rx_meas)
rx_meas2 = rx_meas2/max(rx_meas2)
rx_meas3 = rx_meas3/max(rx_meas3)
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
w, H = functions.channel_est(psi_orth = rx_meas, y_cx_received = rx_meas3)

x_canc = np.squeeze(np.dot(H,w)) # np.squeeze is make shape of x_canc same for later calculation
y_canc = rx_meas2 - x_canc.T

plt.plot(rx)
plt.title('rx The received signal before cancellation')
plt.figure()
plt.plot(rx_meas)
plt.title('rx_meas The received signal before cancellation')

#plt.figure()
#plt.plot(tx)
#plt.title('Loopback signal before antenna')
#plt.figure()
#plt.plot(x_canc)
#plt.title('The cancellation signal')

plt.figure()
plt.plot(y_canc)
plt.title('Remaining received signal after cancellation ')

# Pulse compression after cancellation
PC1 = functions.PulseCompr(rx=y_canc,tx=rx_meas,win=np.blackman(N))
PC1 = LPF_beatfreq(PC1)
PC_log1 = 20*np.log10(abs(PC1))

# Pulse compression before cancellation
PC2 = functions.PulseCompr(rx = rx_meas3,tx = rx_meas,win = np.blackman(N))
PC2 = LPF_beatfreq(PC2) # added
PC_log2 = 20*np.log10(abs(PC2))
plt.figure()
plt.plot(distance, PC_log1, 'k*-', distance,PC_log2, 'b')
plt.xlim((-100,100))
plt.ylim((-50,50))
plt.title('Pulse Compression')
plt.xlabel('Distance in meter')
plt.ylabel('Power in dB')
plt.legend(['After Leakage Cancellation','Before Leakage Cancellation'])
plt.grid()

'''
# Pulse compression with just cancellation signal
PC3 = functions.PulseCompr(rx = x_canc.T,tx=tx,win=np.blackman(N))
PC3 = LPF_beatfreq(PC3)
PC_log3 = 20*np.log10(abs(PC3))
#plt.figure()
plt.plot(distance,PC_log3,'r')
#plt.xlim((-10,800))
plt.xlabel('Distance in meter')
plt.ylabel('Power in dB')
plt.title('Pulse compression with Cancellation Signal')
plt.grid()
'''
plt.show()