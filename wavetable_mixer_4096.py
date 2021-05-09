from scipy import io as sio
from scipy.signal import hilbert
import numpy as np
#from scipy.signal.waveforms import chirp
import matplotlib.pyplot as plt
import os
import struct

def readbin1(filename, nsamples, fsize, rubish):
    bytespersample = 4
    samplesperpulse = nsamples
    total_samples = fsize / bytespersample
    total_pulse = total_samples / samplesperpulse
    file = open(filename, 'rb')
    file.seek((total_pulse - 1) * bytespersample * samplesperpulse + bytespersample * rubish)  # find the last pulse
    x = file.read(bytespersample * samplesperpulse)
    file.close()
    fmt = ('%sh' % (len(x) / 2))  # e.g.  '500h' means 500 shorts
    x_sig = np.array(struct.unpack(fmt, x)).astype(float)  # convert to complex float
    rx_sig = -x_sig[0::2] + 1j * x_sig[
                                 1::2]  # Important! Here I added a negtive sign here for calibrate the tx rx chain. There is a flip of sign somewhere but I cannot find.
    rx_sig = rx_sig / 32767.0
    return rx_sig
def last_pulse(filename, nsamples, rubish):
    fsize = int(os.stat(filename).st_size)
    signal = readbin1(filename, nsamples, fsize, rubish)
    signal_ch0, signal_ch1 = get_slice(signal, 256)
    return signal_ch0, signal_ch1
def read_last_pulse(filename, N):
    nsamples = N
    channels = 2
    rubish = 0
    # Read latest pulse
    data_ch0, data_ch1 = last_pulse(filename, nsamples * channels, rubish)
    rx_error = np.array(data_ch0[0:N])
    return rx_error
def get_slice(data, pulselenth):
    data_ch0 = []
    data_ch1 = []
    for i in range(0, len(data), 2 * pulselenth):
        data_ch0[i:i + pulselenth] = data[i:i + pulselenth]
        data_ch1[i:i + pulselenth] = data[i + pulselenth:i + 2 * pulselenth]
    return data_ch0, data_ch1

#############
#Parameters
#############
c = 3e8
j = 1j
fs = 250e6 # Sampling freq
N = 4000 #4096-1 # Beacuse in C++ file, it added one more point at end for coeffient.  # This also limit the bandwidth. And this is determined by fpga LUT size.
n = 1
N1 = N*n
T = N/fs  # T=N/fs#Chirp Duration
print (T)
t = np.linspace(0, T, N)
nn = np.linspace(0, N-1, N)
# f0=-28e6#Start Freq
# f1=28e6#End freq
f0 = 29.5e6  # Start Freq
f1 = 30.5e6 # fs/2=1/2*N/T#End freq
k = (f1 - f0) / T
phi0 = -(4999+1) * np.pi / 10000  # Phase
phi_init = 0/180.0 * np.pi  # Initial Phase
amp0 = -0.45
f = np.linspace(0, fs-1, N)
freq = np.fft.fftfreq(N1, d=1. / fs)
win=np.blackman(N)
#win=np.hamming(N)
win=1

##################
# Create the chirp
##################
y0 = amp0*np.sin(phi_init + 2 * np.pi * (f0 * t + k / 2 * np.power(t, 2)))  # use this for chirp generation
yq0 = amp0*np.sin(phi_init + phi0 + 2 * np.pi * (f0 * t + k / 2 * np.power(t, 2)))  # use this for chirp generation


#y0 = amp0*np.sin(2 * np.pi * (f0 * t ))  # use this for chirp generation
#yq0 = amp0*np.sin(phi0 + 2 * np.pi * (f0 * t ))

#y0 = amp0*np.sin(2*np.pi*1.0*fs/N*t+phi_init)#+ np.sin(4*np.pi*fs/N*t)# just use LO to generate a LO. The
#yq0 = amp0*np.sin(2*np.pi*1.0*fs/N*t-np.pi/2+phi_init)# + np.sin(4*np.pi*fs/N*t-np.pi/2)

#y0 = np.concatenate((np.zeros(2048),np.ones(2048)))
#yq0 = np.concatenate((np.zeros(2048),np.ones(2048)))

#y0 = np.linspace(-1,1,N)
#yq0 = np.linspace(1,-1,N)


#y_hb = hilbert(y0)#y0 * np.exp(j * 1 / 2 * np.pi)  # np.real (np.fft.ifft(np.exp(-j*2*np.pi*f*(1/f1)/4*fs/fs)*(np.fft.fft(y))))#j*yq #
#y0 = np.concatenate( (np.zeros(N / 2), np.ones(N / 2)), axis = 0)
#yq0 = np.zeros(N)

#y0 = 0.5*np.ones(N)
#yq0 = 0.5*np.ones(N)
#y0 = np.sin(2*np.pi*1e3*t)# simple sine wave
#yq0 = np.sin(2*np.pi*1e3*t-np.pi/2)
#y0 = np.zeros(N)
#suo yq0 = np.zeros(N)

mean = 0
std = 1/np.sqrt(2)/2
noise1 = np.random.normal(mean, std, size=N)
#y0 = noise1
#yq0 = noise1
#noise2 = np.random.normal(mean, std, size=N)
#noise1 = 1*std * np.sin(2*np.pi*6.45e6*t)
#y_noise1 = np.int16(1/32767.0*np.multiply(np.round(32767*win), np.round((32767*(1-2*std)*(np.multiply(y0, 1)+noise1)))))#good
#yq_noise1 = np.int16(1/32767.0*np.multiply(np.round(32767*win), np.round((32767*(1-2*std)*(np.multiply(yq0, 1)+noise1)))))
#y_noise2 = np.int16(np.round((32767*(1-2*std)*(np.multiply(y0, 1)+noise2))))#good
#yq_noise2 = np.int16(np.round((32767*(1-2*std)*(np.multiply(yq0, 1)+noise2))))

y = np.around(32767*np.multiply(y0, win))
yq = np.around(32767*np.multiply(yq0, win))

#y_hb_int = np.around(np.power(2,15)*np.multiply(y_hb, win))
# y=np.sin(2*np.pi*4e6*t)+np.sin(2*np.pi*10e6*t)
# yq=np.sin(2*np.pi*4e6*t-np.pi/2)+np.sin(2*np.pi*10e6*t-np.pi/2)

#
#y_cx_can1 = np.transpose(sio.loadmat('canP4_1024.mat').get('r2')) # 1023 points chapter 4
#y_cx_can1 = y_cx_can1/np.max(y_cx_can1)
#y_cx_can = np.append(y_cx_can1, 0)
#N/(j*2*np.pi*f0)
#y_cx_0 = np.multiply(y0, 1) + j*np.multiply(yq0, 1)
#y_cx_2 = np.multiply(y0, win) + j*np.multiply(yq0, win)
y_cx_1 = y + j * yq
#y_cx_noise1 = y_noise1 + j * yq_noise1
#y_cx_noise2 = y_noise2 + j * yq_noise2
#y_cx_delay  = np.concatenate((np.zeros(200), y_cx_1[:N-200]), axis=0)#y_cx_1 #np.multiply(yq, win), y_cx_0 or y_cx_can
y_cx = 1*y_cx_1
#y_cx = np.multiply(y_c*32767x_can, win)
#y_cx = np.multiply(y_cx_0, win)
y_cx_long = []
for i in range(0, n):
    y_cx_long = np.append(y_cx_long, y_cx)
y_cx = y_cx_long
######################
# Test TX Waveform
######################
'''
y_cx [1] = y_cx [2]
y_cx [0] = y_cx [1]

y_cx [64] = y_cx [65]
y_cx [63] = y_cx [64]
y_cx [128] = y_cx [129]
y_cx [127] = y_cx [128]

y_cx [192] = y_cx [193]
y_cx [191] = y_cx [192]

y_cx [254] = y_cx [255]y_cx_delay
y_cx [253] = y_cx [254]'''
y_cx_delay_1 = np.concatenate((y_cx[N-20:N],y_cx[0:N-20]),axis = 0)
#y_cx_delay_1np.concatenate((np.zeros(50), y_cx[:N-50]), axis=0))
#y_cx_delay_2 = np.concatenate((np.zeros(6000), y_cx[:N-6000]), axis=0)
y_cx_delay = y_cx_delay_1#y_cx # y_cx_delay_1#y_cx#-y_cx_delay_1# y_cx#-y_cx_delay_1#1 * y_cx #+ 1*y_cx_delay_1 + 1*y_cx_delay_2


'''
##########################
# Pre-distortion
##########################
e = np.reshape(read_last_pulse("usrp_samples_loopback.dat", N), [N, 1])
E_inv = np.squeeze(1./np.fft.fft(e, axis=0 ), axis=1)
#E_inv = 1./(np.fft.fft(y_cx, axis=0 ) + 1e-9)
X0 = np.fft.fft(y_cx, axis=0 )
Xp0 = np.multiply(np.multiply(X0, X0), E_inv)
xp0 = np.around(np.fft.ifft(Xp0, axis=0)/6.4e9*32767)
y_cx_delay = xp0
plt.plot(xp0.real)
plt.plot(xp0.imag)
'''

#############################
# Pulse Compression
#############################
# Using Mixer
A = np.multiply(y_cx,np.blackman(N))
B = np.multiply(y_cx_delay,np.blackman(N))
print("A is" , len(A), len(B))
PC = 20*np.log10(abs(np.fft.fft(A*np.conj(B))))
PC = PC - max(PC)
#PC = 20*np.log10(abs(np.fft.fft(np.real(A) * np.real(B))))
#PC = PC-np.max(PC) #normalization
# Match Filtering
#A = y_cx#(np.fft.fft(y_cx_delay))#SIG #np.fft.fft(y_cx_long)
#B = y_cx #np.conj(np.fft.fft(y_cx)) #np.conj(np.fft.fft(np.real(y_cx)+j*np.imag(y_cx)))
MF = np.correlate(A, (B),"same")
#A = np.fft.fft(y_cx)#SIG #np.fft.fft(y_cx_long)
#B = y_cx #np.conj(np.fft.fft(y_cx)) #np.conj(np.fft.fft(np.real(y_cx)+j*np.imag(y_cx)))
#MF = np.multiply(A,B)
'''MF_ac = np.multiply(np.real(A), np.real(B))
MF_bd = np.multiply(np.imag(A), np.imag(B))
MF_real = np.multiply(np.real(A), np.real(B)) - np.multiply(np.imag(A), np.imag(B))   #20*np.log10(np.abs(np.fft.ifft((np.multiply(A, B)))))
MF_imag = np.multiply(np.real(A), np.imag(B)) + np.multiply(np.imag(A), np.real(B))   #20*np.log10(np.abs(np.fft.ifft((np.multiply(A, B)))))
MF = MF_ac -MF_bd#MF_real #+ j * MF_imag'''
A = np.fft.fft(A)
B = np.fft.fft(B)
MF = np.multiply(B, np.conj(A))
MF = np.fft.ifft(MF)
MF = 20 * np.log10(np.abs(MF))
MF = MF-np.max(MF) #normalization

#################
# Plot
#################
#plt.figure(1)
#plt.plot(np.real(y_cx_1))
#plt.plot(t, np.real(y_cx_delay), '*-', t, np.imag(y_cx_delay), 'r*-')
#plt.grid(True)

plt.figure(2)
#plt.plot( np.fft.fftshift(PC))
#plt.plot( freq, np.fft.fftshift(MF), freq, PC)
plt.plot( freq, MF,'*-', freq, PC,'^-')
#plt.figure(3)
#plt.plot(20*np.log(A))
#plt.hold
#plt.plot( t, MF)
plt.grid(True)
print ('the max pc is %f dB' % (max(PC)))

##################
# Save to file
##################
y_cxnew = y_cx_delay#np.multiply(y_cx,win)
yw = np.zeros(2 * N)

for i in range(0, N):
    yw[2 * i + 1] = np.imag(y_cxnew[i]) # tx signal
    yw[2 * i] = np.real(y_cxnew[i])  # tx signal
yw = np.int16(yw)  # E312 setting --type short
#yw = np.float32(yw)  # E312 setting --type float
#yw = np.float64(yw)  # E312 setting --type double
print (max(yw))
data = open('usrp_samples.dat', 'w')
#data.write(yw)
data.close()

plt.show()