import numpy as np
import matplotlib.pyplot as plt
import os
from readbin import readbin2
from tempfile import TemporaryFile
import readbin
import scipy.io as sio

#############
# Parameters
#############
c = 3e8
j = 1j
fs = 28e6  # Sampling freq
N = 4092  # This also limit the bandwidth. And this is determined by fpga LUT size.
tc = N/fs  # T=N/fs#Chirp Duration
t = np.linspace(0, tc, N)
nn = np.linspace(0, N-1, N)
f0 = -14e6  # Start Freq
f1 = 14e6  # fs/2=1/2*N/T#End freq
K = (f1 - f0) / tc # chirp rate = BW/Druation
NFFT =  N
freq = np.fft.fftfreq(NFFT, d=1 / fs)
distance = c * freq / K / 2.0

###################
# Import Signals
###################
# Transmited signal
outfile = "ref_signal_20190410_avg1_28MHz.npy"
tx_sig = np.load(outfile)
#plt.figure()
#plt.plot(t, np.real(tx_sig), '*-' ,t, np.imag(tx_sig) , '*-')
# Received signal
nsamples = N
times =1
rubish = 4096*12
num_of_channel = 2
chirp_length = N
rx_data = readbin.average("usrp_samples_loopback.dat", nsamples, times, rubish, num_of_channel, chirp_length)
rx_ch0, rx_ch1 = rx_data
rx_ch0 = np.array(rx_ch0[0:N])
rx_ch1 = np.array(rx_ch1[0:N])

rx_avg = rx_ch0
####################
# Leakage Cancellation
####################


def channel_est(psi_orth, y_cx_received):
    # Calculate weights
    # 1. Make H matrix
    psi_orth_delay_0 = np.concatenate((np.zeros(0), psi_orth[:N-0]), axis=0)
    psi_orth_delay_1 = np.concatenate((np.zeros(1), psi_orth[:N-1]), axis=0)
    psi_orth_delay_2 = np.concatenate((np.zeros(2), psi_orth[:N-2]), axis=0)
    psi_orth_delay_3 = np.concatenate((np.zeros(3), psi_orth[:N-3]), axis=0)
    psi_orth_delay_4 = np.concatenate((np.zeros(4), psi_orth[:N-4]), axis=0)
    psi_orth_delay_5 = np.concatenate((np.zeros(5), psi_orth[:N-5]), axis=0)
    psi_orth_delay_6 = np.concatenate((np.zeros(6), psi_orth[:N-6]), axis=0)
    psi_orth_delay_7 = np.concatenate((np.zeros(7), psi_orth[:N-7]), axis=0)
    psi_orth_delay_8 = np.concatenate((np.zeros(8), psi_orth[:N-8]), axis=0)
    psi_orth_delay_9 = np.concatenate((np.zeros(9), psi_orth[:N-9]), axis=0)
    H= np.matrix(np.transpose([psi_orth_delay_0,psi_orth_delay_1,psi_orth_delay_2,psi_orth_delay_3,psi_orth_delay_4,psi_orth_delay_5,
        psi_orth_delay_6,psi_orth_delay_7,psi_orth_delay_8,psi_orth_delay_9]))
    #print('Shape of H is',H.shape)

    # 2. Make y vector
    y = np.transpose(y_cx_received[np.newaxis])
    #print('Shape of y is', y.shape)

    # 3. Calculate channel
    R_H = np.dot(np.matrix.getH(H),H) # Covariance Matrix of H
    #print('Shape of R_H is', R_H.shape)
    R_yH = np.dot(np.matrix.getH(H),y)# Cross-covariance Matrix of {y,H}
    #print('Shape of R_yH is', R_yH.shape)
    c = np.dot(np.linalg.inv(R_H),R_yH)
    #print('Shape of c is', c.shape)
    return c, H

######################################
# Pulse Compression Stretching Method
#####################################


def PulseCompr(rx,tx,win):
    A = np.multiply(rx,win) # Add window here
    B = tx
    PC = 20*np.log10(abs(np.fft.fft(A*np.conj(B))))
    return PC


w,H = channel_est(psi_orth = tx_sig, y_cx_received = rx_avg)
x_canc = np.squeeze(np.array(np.dot(H,w))) # np.squeeze is make shape of x_canc same for later calculation
plt.figure()
plt.plot(tx_sig)
plt.title('The transmit signal')
plt.figure()
plt.plot(rx_avg)
plt.title('The received signal')
plt.figure()
plt.plot(x_canc)
plt.title('The cancellation signal')
y_canc = rx_avg - x_canc
plt.figure()
plt.plot(y_canc)
plt.title('Remaining received signal after cancellation ')
# Pulse compression after cancellation
PC = PulseCompr(rx=y_canc,tx=tx_sig,win=np.blackman(N))
plt.figure()
plt.plot(distance,PC,'k',label='After Leakage Cancellation')
#plt.xlim((-10,800))
plt.xlabel('Distance in meter')
plt.ylabel('Power in dB')
plt.grid()
# Pulse compression before cancellation
PC = PulseCompr(rx = rx_avg,tx = tx_sig,win = np.blackman(N))
plt.plot(distance,PC,'b',label='Before Leakage Cancellation')
plt.title('Pulse Compression')
plt.legend()

# Pulse compression with just cancellation signal
PC = PulseCompr(rx = x_canc.T,tx=tx_sig,win=np.blackman(N))
plt.figure()
plt.plot(distance,PC,'r')
#plt.xlim((-10,800))
plt.xlabel('Distance in meter')
plt.ylabel('Power in dB')
plt.title('Pulse compression with Cancellation Signal')
plt.grid()

print(1/N*np.sum(y_canc),w)
plt.show()
