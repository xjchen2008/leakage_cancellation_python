from scipy import io as sio
from scipy.signal import hilbert
import numpy
#from scipy.signal.waveforms import chirp
import matplotlib.pyplot as plt
#############
#Parameters
#############
c = 3e8
j = 1j
fs = 28e6  # Sampling freq
N = 4096+1 # Beacuse in C++ file, it added one more point at end for coeffient.  # This also limit the bandwidth. And this is determined by fpga LUT size.
n = 1
N1 = N*n
T = N/fs  # T=N/fs#Chirp Duration
print (T)
t = numpy.linspace(0, T, N)
nn = numpy.linspace(0, N-1, N)
# f0=-28e6#Start Freq
# f1=28e6#End freq
f0 = -1e6#-14e5  # Start Freq
f1 = 1e6 # fs/2=1/2*N/T#End freq
k = (f1 - f0) / T
phi0 = -(4999+1) * numpy.pi / 10000  # Phase
f = numpy.linspace(0, fs-1, N)
freq = numpy.fft.rfftfreq(N1, d=1. / fs)
win=numpy.blackman(N)
#win=numpy.hamming(N)
win=1

##################
# Create the chirp
##################
y0 = 1*numpy.sin(2 * numpy.pi * (f0 * t + k / 2 * numpy.power(t, 2))) + 0.0 # use this for chirp generation
yq0 = 1*numpy.sin(phi0 + 2 * numpy.pi * (f0 * t + k / 2 * numpy.power(t, 2)))  # use this for chirp generation
#y0 = numpy.sin(2*numpy.pi*fs/N*t)#+ numpy.sin(4*numpy.pi*fs/N*t)# just use LO to generate a LO. The
#yq0 = numpy.sin(2*numpy.pi*fs/N*t-numpy.pi/2)# + numpy.sin(4*numpy.pi*fs/N*t-numpy.pi/2)



#y_hb = hilbert(y0)#y0 * numpy.exp(j * 1 / 2 * numpy.pi)  # numpy.real (numpy.fft.ifft(numpy.exp(-j*2*numpy.pi*f*(1/f1)/4*fs/fs)*(numpy.fft.fft(y))))#j*yq #
#y0 = numpy.concatenate( (numpy.zeros(N / 2), numpy.ones(N / 2)), axis = 0)
#yq0 = numpy.zeros(N)

#y0 = 0.5*numpy.ones(N)
#yq0 = 0.5*numpy.ones(N)
#y0 = numpy.sin(2*numpy.pi*1e3*t)# simple sine wave
#yq0 = numpy.sin(2*numpy.pi*1e3*t-numpy.pi/2)
#y0 = numpy.zeros(N)
#yq0 = numpy.zeros(N)

#mean = 0
#std = 0.18
#noise1 = numpy.random.normal(mean, std, size=N)
#noise2 = numpy.random.normal(mean, std, size=N)
#noise1 = 1*std * numpy.sin(2*numpy.pi*6.45e6*t)
#y_noise1 = numpy.int16(1/32767.0*numpy.multiply(numpy.round(32767*win), numpy.round((32767*(1-2*std)*(numpy.multiply(y0, 1)+noise1)))))#good
#yq_noise1 = numpy.int16(1/32767.0*numpy.multiply(numpy.round(32767*win), numpy.round((32767*(1-2*std)*(numpy.multiply(yq0, 1)+noise1)))))
#y_noise2 = numpy.int16(numpy.round((32767*(1-2*std)*(numpy.multiply(y0, 1)+noise2))))#good
#yq_noise2 = numpy.int16(numpy.round((32767*(1-2*std)*(numpy.multiply(yq0, 1)+noise2))))
y = numpy.around(32767*numpy.multiply(y0, win))
yq = numpy.around(32767*numpy.multiply(yq0, win))
#y_hb_int = numpy.around(numpy.power(2,15)*numpy.multiply(y_hb, win))
# y=numpy.sin(2*numpy.pi*4e6*t)+numpy.sin(2*numpy.pi*10e6*t)
# yq=numpy.sin(2*numpy.pi*4e6*t-numpy.pi/2)+numpy.sin(2*numpy.pi*10e6*t-numpy.pi/2)

#
#y_cx_can1 = numpy.transpose(sio.loadmat('canP4_1024.mat').get('r2')) # 1023 points chapter 4
#y_cx_can1 = y_cx_can1/numpy.max(y_cx_can1)
#y_cx_can = numpy.append(y_cx_can1, 0)
#N/(j*2*numpy.pi*f0)
#y_cx_0 = numpy.multiply(y0, 1) + j*numpy.multiply(yq0, 1)
#y_cx_2 = numpy.multiply(y0, win) + j*numpy.multiply(yq0, win)
y_cx_1 = y# + j * yq
#y_cx_noise1 = y_noise1 + j * yq_noise1
#y_cx_noise2 = y_noise2 + j * yq_noise2
#y_cx_delay  = numpy.concatenate((numpy.zeros(200), y_cx_1[:N-200]), axis=0)#y_cx_1 #numpy.multiply(yq, win), y_cx_0 or y_cx_can
y_cx = 1*y_cx_1
#y_cx = numpy.multiply(y_cx_can, win)
#y_cx = numpy.multiply(y_cx_0, win)
y_cx_long = []
for i in range(0, n):
    y_cx_long = numpy.append(y_cx_long, y_cx)
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

y_cx [254] = y_cx [255]
y_cx [253] = y_cx [254]'''
y_cx_delay_1 = numpy.concatenate((y_cx[N-50:N],y_cx[0:N-50]),axis = 0)
#y_cx_delay_1numpy.concatenate((numpy.zeros(50), y_cx[:N-50]), axis=0))
#y_cx_delay_2 = numpy.concatenate((numpy.zeros(6000), y_cx[:N-6000]), axis=0)
y_cx_delay = y_cx #y_cx_delay_1#1 * y_cx #+ 1*y_cx_delay_1 + 1*y_cx_delay_2

#############################
# Pulse Compression
#############################
# Using Mixer
A = numpy.multiply(y_cx_delay,1)
B = y_cx_delay
print("A is" , len(A), len(B))
PC = 20*numpy.log10(abs(numpy.fft.rfft(A*numpy.conj(B))))
#PC = 20*numpy.log10(abs(numpy.fft.fft(numpy.real(A) * numpy.real(B))))
#PC = PC-numpy.max(PC) #normalization
'''
# Match Filtering
A = (numpy.fft.fft(y_cx_delay))#SIG #numpy.fft.fft(y_cx_long)
B = numpy.conj(numpy.fft.fft(y_cx)) #numpy.conj(numpy.fft.fft(numpy.real(y_cx)+j*numpy.imag(y_cx)))
#MF_ac = numpy.multiply(numpy.real(A), numpy.real(B))
#MF_bd = numpy.multiply(numpy.imag(A), numpy.imag(B))
#MF_real = numpy.multiply(numpy.real(A), numpy.real(B)) - numpy.multiply(numpy.imag(A), numpy.imag(B))   #20*numpy.log10(numpy.abs(numpy.fft.ifft((numpy.multiply(A, B)))))
#MF_imag = numpy.multiply(numpy.real(A), numpy.imag(B)) + numpy.multiply(numpy.imag(A), numpy.real(B))   #20*numpy.log10(numpy.abs(numpy.fft.ifft((numpy.multiply(A, B)))))
#MF = MF_ac -MF_bd#MF_real #+ j * MF_imag
MF = numpy.multiply(A, B)
MF = numpy.fft.ifft(MF)
MF = 20 * numpy.log10(numpy.abs(MF))
MF = MF-numpy.max(MF) #normalization
'''
#################
# Plot
#################
plt.figure(1)
#plt.plot(numpy.real(y_cx_1))
plt.plot(t, A)
plt.grid(True)

plt.figure(2)
plt.plot( freq, (PC))
plt.figure(3)
plt.plot(20*numpy.log(A))
#plt.hold
#plt.plot( t, MF)
plt.grid(True)
print ('the max pc is %f dB' % (max(PC)))

##################
# Save to file
##################
y_cxnew = y_cx_delay#numpy.multiply(y_cx,win)
yw = numpy.zeros(2 * N)

for i in range(0, N):
    yw[2 * i + 1] = numpy.imag(y_cxnew[i]) # tx signal
    yw[2 * i] = numpy.real(y_cxnew[i])  # tx signal
yw = numpy.int16(yw)  # E312 setting --type short
#yw = numpy.float32(yw)  # E312 setting --type float
#yw = numpy.float64(yw)  # E312 setting --type double

data = open('usrp_samples.dat', 'w')
data.write(yw)
data.close()

plt.show()