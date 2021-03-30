from readbin import readbin2
from matplotlib import pyplot as plt
import numpy as np

N=4096*500#1120000*2
rubish = 0 #256*8
#sig = readbin2("/home/james/rfnoc/src/uhd/host/examples/replay_samples_from_file/build/usrp_samples_loopback.dat", np.short,N)
sig = readbin2("usrp_samples_loopback.dat", np.short,N,rubish)
#sig1 = readbin2("usrp_samples_loopback_thermal_10232018_1.dat", np.short, N)
#sig2 = readbin2("usrp_samples_loopback_thermal_10232018_2.dat", np.short, N)
#sig3 = readbin2("usrp_samples_loopback_thermal_10232018_3.dat", np.short, N)
#x = np.linspace(0, N-1, N)
#plt.plot(sig, 'r-',marker="*")
#plt.figure(2)
#freq = np.fft.fftfreq(N)
#SIG = 20*np.log10(np.fft.fft(sig))
#plt.plot(freq, SIG)
#plt.figure(1)
#plt.plot(x, np.real(sig1), x,np.real(sig2), x, np.real(sig3))
#plt.figure(2)
#plt.plot(x, np.imag(sig1), x,np.real(sig1))
#power1 = np.sum(pow(np.abs(np.real(sig1)),2))
#power2 = np.sum(pow(np.abs(np.real(sig2)),2))
#power3 = np.sum(pow(np.abs(np.real(sig3)),2))
#print power1, power2, power3
#sig2_cal = power1/power2*sig2
#sig3_cal = power1/power3*sig3
#plt.figure(1)
#plt.plot(x, 32767*np.imag(sig), x,32767*np.real(sig))
index = 0
'''
for i in range(3*256, 4*256):
    if 32767*np.imag(sig[i]) >= 32766:
        index = i
    if 32767*np.imag(sig[i]) <= 32766 and i >= index and index is not 0:
        sig = np.delete(sig, np.s_[0:i-1])
        break
'''
#x = np.linspace(0, len(sig)-1, len(sig))
#plt.plot(x, 32767*np.imag(sig),x, 32767*np.real(sig))
#plt.grid()



def get_slice(data, pulselenth):
    data_ch0 = []
    data_ch1 = []
    for i in range(0,len(data), 2*pulselenth):
        data_ch0[i:i+pulselenth] = data[i:i+pulselenth]
        data_ch1[i:i+pulselenth] = data[i+pulselenth+1:i + 2*pulselenth]
    return data_ch0, data_ch1

#plt.figure(2)
data_ch0, data_ch1 = get_slice(sig,256)
plt.plot(np.real(data_ch0),'*-')
plt.plot(np.imag(data_ch0),'*-')
plt.grid()

plt.figure(3)
plt.plot(np.real(data_ch1),'*-')
plt.plot(np.imag(data_ch1),'*-')

plt.show()