import functions
import numpy as np
import matplotlib.pyplot as plt
import readosc
import coe_wavetable_4096 as coe
import functions
import dsp_filters
from scipy import signal
from numpy.fft import fftshift
import dsp_filters_BPF
import setup


def LPF_beatfreq(pc):
    pc_timedomain = np.fft.ifft(pc)
    pc_timedomain_LPF = dsp_filters.main(signal=pc_timedomain, order=6, fs=fs, cutoff=50e6, duration=T)
    pc_timedomain_LPF_win = np.multiply(pc_timedomain_LPF, np.blackman(len(pc)))  # .reshape([N,1]))
    pc_freqdomain_LPF = np.fft.fft(pc_timedomain_LPF_win)
    # pc_log = 20 * np.log10(abs(pc_freqdomain_LPF))  # with LPF for sretch method
    # pc_log = 20 * np.log10(abs(pc))# no LPF
    #pc_log = pc_log - max(pc_log)  # normalization
    return pc_freqdomain_LPF

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
freq = np.fft.fftfreq(N*setup.upsamp_rate, d=1 / fs/setup.upsamp_rate)
distance = c * freq / K / 2.0
win = np.blackman(N*setup.upsamp_rate)


tx = functions.upsampling(coe.y_cx.real, setup.upsamp_rate) #readosc.readcsv(filename=setup.simulation_filename) #coe.y_cx.real #np.load(file=setup.file_tx)
tx = tx.real
#tx = signal.hilbert(tx)

def read_rx(filename):
    rx_meas3 = readosc.readcsv(filename=filename)
    rx_meas3 = functions.upsampling(rx_meas3, setup.upsamp_rate)
    #rx_meas3 = (rx_meas3-np.mean(rx_meas3))#/max(rx_meas3)
    #rx_meas3 = signal.hilbert(rx_meas3)
    #rx_meas3 = functions.upsampling(rx_meas3, 1)
    RX_meas3 = np.fft.fft(rx_meas3)
    RX_meas3[0:100] =0  # set the dc part of the rx signal to 0.
    RX_meas3[-1-100:] = 0
    rx_meas3 = np.fft.ifft(RX_meas3)
    rx_meas3 = rx_meas3/max(rx_meas3) # Normalized after setting dc to 0.
    #rx_meas3 = dsp_filters_BPF.run(rx_meas3)
    rx_meas3 = rx_meas3.real
    return rx_meas3


step = 2
if step == 2:
    # Read rx signal
    filename = 'BPF_Antenna_499999_indoor_40_60MHz_chirp_withPA_antialiasLPF_0529_6.csv'  #'data/avg/antenna_499999_indoor_40_60MHz_chirp_Noavg_measure_afterCanc2_D100_delaym40_ch13997.csv'
    #filename = 'BPF_Antenna_3999_indoor_40_60MHz_chirp_N100avg_withPA_0516_nocanc.csv'
    rx_meas3 = read_rx(filename=filename)
    x_canc = 1*np.load(file='x_canc_PA.npy')
if step == 1:
    # Read rx signal
    filename = 'BPF_Antenna_499999_indoor_40_60MHz_chirp_withPA_antialiasLPF_0529_0.csv' #'data/avg/antenna_499999_indoor_40_60MHz_chirp_Noavg_measure_afterCanc2_D100_delaym40_ch13999.csv'
    rx_meas3 = read_rx(filename=filename)
    # Generate cancellation signal
    w, H = functions.channel_est(psi_orth = tx, y_cx_received = rx_meas3)
    x_canc = np.squeeze(np.array(np.dot(H,w))) # np.squeeze is make shape of x_canc same for later calculation
    np.save('x_canc_PA', x_canc)
if step == 3:
    filename1 = 'BPF_Antenna_499999_indoor_40_60MHz_chirp_withPA_antialiasLPF_0529_0.csv'
    rx_meas1 = read_rx(filename=filename1)
    filename2 = 'BPF_Antenna_499999_indoor_40_60MHz_chirp_withPA_antialiasLPF_0529_4.csv'
    rx_meas2 = read_rx(filename=filename2)
    rx_meas3 = rx_meas1- rx_meas2

    w, H = functions.channel_est(psi_orth=tx, y_cx_received=rx_meas3)
    x_canc = np.squeeze(np.array(np.dot(H,w))) #np.zeros(N)

y_canc = rx_meas3 - np.roll(x_canc.T,0)
#y_canc = dsp_filters.main(signal=y_canc, order=6, fs=fs, cutoff=60e6, duration=T)
#plt.plot(rx)
#plt.title('rx The received signal before cancellation')
plt.figure()
plt.plot(rx_meas3)
#plt.title('rx_meas3 The received signal before cancellation')

#plt.figure()
#plt.plot(tx)
#plt.title('Loopback signal before antenna')
#plt.figure()
plt.plot(x_canc)
plt.title('The cancellation signal')
plt.legend(['The received signal before canellation','The cancellation signal'])

plt.figure()
plt.plot(y_canc)
plt.title('Remaining received signal after cancellation ')
# Frequency response
plt.figure()
functions.plot_freq_db(freq / 1e6, rx_meas3.real, color='b')
# plt.figure()
functions.plot_freq_db(freq / 1e6, y_canc.real, color='k')
plt.xlabel('Frequency [MHz]')
plt.ylabel('Amplitude [dB]')
#plt.ylim(-50, 50)


# Pulse compression after cancellation
win =  np.blackman(N*setup.upsamp_rate)
PC1 = functions.PulseCompr(rx=signal.hilbert(y_canc.real),tx=signal.hilbert(tx),win=win)

# Pulse compression before cancellation
PC2 = functions.PulseCompr(rx = signal.hilbert(rx_meas3),tx = signal.hilbert(tx),win = win)
plt.figure()
plt.plot(fftshift(distance), fftshift(PC1), 'k*-', fftshift(distance),fftshift(PC2), 'b')
#plt.xlim((-100,200))
#plt.ylim((-90,100))
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