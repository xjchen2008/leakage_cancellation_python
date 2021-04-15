#from readbin import readbin2, average
import scipy
import numpy
from scipy.signal.waveforms import chirp
import matplotlib.pyplot as plt

#############
#Parameters
#############
c = 3e8
j = 1j
fs = 250e6  # Sampling freq

N = 5000  # This also limit the bandwidth. And this is determined by fpga LUT size.
T = N/fs  # T=N/fs#Chirp Duration
print (N)
t = numpy.linspace(0, T, N)
bw = 20.0e6
fc= 30e6
f0 = fc-bw/2#-10e6#40e6 # Start Freq
f1 = fc+bw/2#10e6#60e6# fs/2=1/2*N/T#End freq
k = (f1 - f0) / T
phi0 = -numpy.pi / 2  # Phase
#f = 1 / 16e6 / 4
freq = numpy.fft.fftfreq(N, d=1. / fs)
#win=numpy.blackman(N)
#win=numpy.hamming(N)
win=1
##################
# Create the chirp
##################
y = numpy.sin(2 * numpy.pi * (f0 * t + k / 2 * numpy.power(t, 2)))  # use this for chirp generation
yq = numpy.sin(phi0 + 2 * numpy.pi * (f0 * t + k / 2 * numpy.power(t, 2)))  # use this for chirp generation
##################
# Create the sine
##################
y_s = numpy.sin(10*2*numpy.pi*fs/N*t)#+ numpy.sin(4*numpy.pi*fs/N*t)# just use LO to generate a LO. The
yq_s = numpy.sin(10*2*numpy.pi*fs/N*t-numpy.pi/2)# + numpy.sin(4*numpy.pi*fs/N*t-numpy.pi/2)
y_cx_sine = y_s + j * yq_s

y_cx_0 = y + j * yq
y_cx_0_delay = numpy.concatenate((numpy.zeros(100), y_cx_0[:N-100]), axis=0)
#y_cx = 0.5*y_cx_0 + 0.5*y_cx_0_delay
y_cx = y_cx_0 #y_cx_sine

####################################################################################
# Add Window Function for COE file as Reference Template for Receiving Match Filter
####################################################################################
#win=numpy.blackman(N)
#win=numpy.hamming(N)
#win=1
#sig = readbin2("usrp_samples_loopback_chirp_16MHz.dat", numpy.short,N)
#sig[0] = 0  # delete the first element because it is a special point, too big
sig = y_cx
'''
SIG = sig #numpy.fft.fft(numpy.multiply(	sig, win)) # window!!!
SIG = SIG / numpy.amax(SIG) #numpy.conj(SIG / numpy.amax(SIG))  # conjugate of the reference

##################
# Save to file
##################
SIG_COE_int = numpy.zeros(2 * N)
SIG_PLOT = numpy.ones(N) + j * numpy.ones(N)  # a complex number
for i in range(0, N):
    SIG_COE_int[2*i] = numpy.int(32766 * numpy.real(SIG[i]))  # fpga 1024
    SIG_PLOT_R = SIG_COE_int[2*i]
    SIG_COE_int[2*i+1] = numpy.int(32766 * numpy.imag(SIG[i]))  # fpga 1024
    SIG_PLOT_I = SIG_COE_int[2*i+1]
    SIG_PLOT[i] = SIG_PLOT_R + j*SIG_PLOT_I
#############################
# Check
#############################
plt.figure(1)
plt.plot(freq, 20*numpy.log10(numpy.abs(numpy.fft.fft(SIG_PLOT))), 'r-',marker="*")
plt.grid(True)
plt.figure(2)
plt.plot(sig, 'r-',marker="*") # template in distributed mem
plt.grid(True)
plt.figure(3)
'''
#############################
# Match Filtering
#############################
'''
A = y_cx#!!!
B = numpy.conj(SIG_PLOT)# Frequency domain of reference signal. Conjugate of tx has already been counted in previous line.
plt.plot(freq, 20*numpy.log10(numpy.abs(numpy.fft.fft((numpy.multiply(A, B))))))
'''
######################################################################
# It aims to transfer decade data to binary data and stores in a file.
# reference: decade data.
# width: binary bit width.
# Improvement: define the file name.
#######################################################################
'''
width = 16
#reference = YW_COE_int
reference = SIG_COE_int

if len("{0:b}".format(int(max(abs(reference))))) > width:
    print ('The data width is not bigger enough.')
L = len(reference)
fd = open('data.coe', 'w')
for k in range(1, L):
    if k % 2 == 1:  # even numbers
        if reference[k] >= 0:
            temp = "{0:{fill}{width}b}".format(int(reference[k]), fill='0', width=width)
        else:
            temp = "{0:{fill}{width}b}".format(int(pow(2, width) - abs(reference[k])), fill='0', width=width)
        temp0 = temp
    else:
        if reference[k] >= 0:
            temp = "{0:{fill}{width}b}".format(int(reference[k]), fill='0', width=width)
        else:
            temp = "{0:{fill}{width}b}".format(int(pow(2, width) - abs(reference[k])), fill='0', width=width)
        temp = temp0 + temp
        print (temp)
        fd.write(temp + '\n')
fd.close()
'''
#plt.show()