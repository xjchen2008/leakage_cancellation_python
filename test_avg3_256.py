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
N = 256  # This also limit the bandwidth. And this is determined by fpga LUT size.
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
times =5
rubish = 256*10
num_of_channel = 2
chirp_length = N
rx_avg = readbin.average("usrp_samples_loopback.dat", nsamples, times, rubish, num_of_channel, chirp_length)


data_ch0, data_ch1 = rx_avg
#plt.figure()
#plt.plot(np.real(data_ch0),'*-')
#plt.plot(np.imag(data_ch0),'*-')
#plt.figure()
#plt.plot(np.real(data_ch1))
#plt.plot(np.imag(data_ch1))

#rx_sig = readbin2("usrp_samples_loopback.dat", np.short,N)
data_ch1 = data_ch0[0:N]

savefile = "ref_signal_20190410_avg1_28MHz.npy"
np.save(savefile, data_ch1)

######################################
# Pulse Compression Stretching Method
#####################################


def PulseCompr(rx,tx,win):
    A = np.multiply(rx,win) # Add window here
    B = tx
    PC = 20*np.log10(abs(np.fft.fft(A*np.conj(B), n = NFFT )))
    #PC = 20 * np.log10(abs(np.fft.ifft(np.fft.fft(A) * np.conj(np.fft.fft(B)), n=NFFT)))
    #PC = 20 * np.log10(abs(np.fft.rfft(np.real(A) * np.real(B), n=NFFT)))
    return PC


PC = PulseCompr(rx = data_ch1,tx = tx_sig,win=np.blackman(len(tx_sig)))


plt.figure()
plt.plot(nn, np.real(data_ch1), '*-',nn, np.imag(data_ch1), '*-')
plt.grid()

plt.figure()
plt.plot(distance,PC -np.max(PC))
#plt.xlim((-10,800))
#plt.ylim((-100,10))
plt.xlabel('Distance in meter')
plt.ylabel('Power in dB')
plt.grid()

plt.figure()
plt.plot(freq, 20*np.log(np.fft.fft(data_ch1)))
plt.grid()

data_ch1_deleted = np.delete(data_ch1,[57,58,117,118,177,178, 237, 238])
plt.figure()
plt.plot(data_ch1_deleted)

plt.figure()
plt.plot(20*np.log(np.fft.fft(data_ch1_deleted)))
plt.grid()
plt.show()
