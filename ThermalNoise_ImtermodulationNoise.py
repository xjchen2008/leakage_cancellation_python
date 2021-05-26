# G. T. Zhou and J. S. Kenney, "Predicting spectral regrowth of nonlinear power amplifiers," in IEEE Transactions on Communications, vol. 50, no. 5, pp. 718-722, May 2002, doi: 10.1109/TCOMM.2002.1006553.
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fftshift
import functions as fn
from scipy import signal


N = 1024 #100000
win= 1 #np.blackman(N)
coe = fn.Coe(fc=50e6, bw=20e6, fs=250e6, N=N) # Call the class. play with the params here!
freq= coe.freq
distance = coe.distance
x_cx = coe.y_cx
#x_cx = np.around(np.power(2,15) * x_cx) # simulate the quantization noise in FPGA fixed point

x = coe.y_cx.real #+ coe.y_noise.real # received signal template
tx = x_cx #+ coe.y_noise # template signal for transmit
#x_cx = signal.hilbert(coe.y_cx.real, axis = 0)
#x_cx = x_cx * coe.y_cx_sine2

####################################################
d1 = 10 #9 # What happened to pulse compression if change the delay from 10 to 9. The phase discontinuity will smear the energy to other frequency. https://www.keysight.com/us/en/assets/7018-06760/application-notes/5952-8898.pdf?success=true
d2 = 1000
d3 = 300
y_delay_1 = np.concatenate((x[N-d1-1:-1], x[:N-d1]), axis=0)  # What happened to pulse compression if change the delay from 10 to 9. The phase discontinuity will smear the energy to other frequency.
y_delay_2 = np.concatenate((x[N-d2-1:-1], x[:N-d2]), axis=0)
y_delay_3 = np.concatenate((x[N-d3-1:-1], x[:N-d3]), axis=0)
vin1 = 1e0*y_delay_1 + 1e-7*y_delay_2 + 1e-5*y_delay_3  # With leakage, no intermodulatoin #1*y_cx_delay_1 + 1e-7*y_cx_delay_2 + 1e-5*y_cx_delay_3
vin2 = 1e-7*y_delay_2 + 1e-5*y_delay_3 # No leakage signal
vin2 = vin2 - 0.01*np.power(vin2,3) \
       + 0.01 * np.power(vin2, 2) \
       + 0.001 * np.power(vin2, 4) \
       + 0.001 * np.power(vin2, 5) \
       + 0.001 * np.power(vin2, 6) \
       + 0.001 * np.power(vin2, 7) \
       + 0.001 * np.power(vin2, 8) \
       + 0.001 * np.power(vin2, 9)# Intermodulation of vin2
vin3 = vin1 - 0.01*np.power(vin1,3) \
       + 0.01*np.power(vin1,2) \
       + 0.001*np.power(vin1,4) \
       + 0.001*np.power(vin1,5) \
       + 0.001*np.power(vin1,6) \
       + 0.001*np.power(vin1,7) \
       + 0.001*np.power(vin1,8) \
       + 0.001*np.power(vin1,9)  #  Intermodulation of vin1 with leakage
# first delay, then recorded in Oscilloscope and go to complex signals
vin1 = signal.hilbert(vin1)
vin2 = signal.hilbert(vin2)
vin3 = signal.hilbert(vin3)
# vin2 = y_cx_received # With leakage signal

V0 = 20*np.log10(np.fft.fft(x)) #np.fft.fft(x_cx)#20*np.log10(np.fft.fft(x_cx))
V1 = 20*np.log10(np.fft.fft(vin1)) #20*np.log10(np.fft.fft(vin1))
V2 = 20*np.log10(np.fft.fft(vin2)) #np.fft.fft(vin2)#20*np.log10(np.fft.fft(vin2))
V3 = 20*np.log10(np.fft.fft(vin3)) #np.fft.fft(vin3)#20*np.log10(np.fft.fft(vin3))
plt.plot(fftshift(freq)/1e6, fftshift(V0-max(V0)))
plt.grid()
plt.title('Original Signal')
plt.figure()
plt.plot(fftshift(freq)/1e6, fftshift(V1-max(V1)),'b*-', fftshift(freq)/1e6, fftshift(V2-max(V2)),'k', fftshift(freq)/1e6, fftshift(V3-max(V3)),'r')
#plt.plot(freq/1e6, vin1,'b', freq/1e6, vin2,'k', freq/1e6, vin3,'r')
plt.title(' Intermodulated Chirp in Frequency Domain')
plt.xlabel('Frequency [MHz]')
plt.ylabel('Normalized Amplitude [dB]')
plt.grid()
plt.legend(['No Intermodulation; leakge signal exists',
           'Intermodulation; without leakage signal',
           'Intermodulation; leakage signal exists'], loc='best')
######################################
# Pulse Compression Stretching Method
#####################################


PC = fn.PulseCompr(rx=vin1,tx=tx,win=np.blackman(N))
PC_intermod = fn.PulseCompr(rx=vin2,tx=tx,win=np.blackman(N))
PC_intermod_leakage = fn.PulseCompr(rx=vin3,tx=tx,win=np.blackman(N))
#f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
#ax1.plot(np.fft.fftshift(PC))
#ax1.set_xlabel('Number of points')
#ax1.set_ylabel('Power in dB')
#ax2.plot(distance,PC)
#ax2.set_title('Pulse Compression Zoom In')
#ax2.set_xlim((-10,500))
#ax2.set_xlabel('Distance in meter')
#ax2.set_ylabel('Power in dB')
#plt.close()



def distance2freq(distance):
    # https://matplotlib.org/stable/gallery/subplots_axes_and_figures/secondary_axis.html
    k = coe.k
    c = 3e8
    freq = distance / c * k * 2.0
    return freq / 1e6  # MHz


def freq2distance(freq):
    k = coe.k
    c = 3e8
    distance = c * freq * 1e6 / k / 2.0
    return distance

fig, ax = plt.subplots()
#plt.plot(distance/1e3,PC-max(PC), '--', distance/1e3,PC_intermod-max(PC), 'k-', distance/1e3, PC_intermod_leakage-max(PC), 'r.-')
#ax.plot(fftshift(distance), fftshift(PC), 'b*-', fftshift(distance), fftshift(PC_intermod), 'k', fftshift(distance), fftshift(PC_intermod_leakage), 'r') # no Normalized
#ax.plot(fftshift(distance), fftshift(PC_intermod-max(PC_intermod)), 'k') #, fftshift(distance), fftshift(PC_intermod-max(PC)), 'k', fftshift(distance), fftshift(PC_intermod_leakage-max(PC)), 'r') # Normalized
#ax.plot(fftshift(distance), fftshift(PC-max(PC)), 'b*-', fftshift(distance), fftshift(PC_intermod-max(PC)), 'k', fftshift(distance), fftshift(PC_intermod_leakage-max(PC)), 'r') # Normalized
ax.plot(fftshift(distance), fftshift(PC_intermod_leakage-max(PC_intermod_leakage)), 'r') #, fftshift(distance), fftshift(PC_intermod-max(PC)), 'k', fftshift(distance), fftshift(PC_intermod_leakage-max(PC)), 'r') # Normalized
plt.xlabel('Distance [m]')
secax = ax.secondary_xaxis('top', functions=(distance2freq, freq2distance))
secax.set_xlabel('Frequency [MHz]')
#ax.set_xlim([-500,2000])
#ax.set_ylim([-80, 20])
#ax.set_ylim([-320, 120])

plt.grid()
plt.legend(['No Intermodulation; leakge signal exists',
           'Intermodulation; without leakage signal',
           'Intermodulation; leakage signal exists'], loc='lower center')

plt.title('Pulse Compression')
plt.ylabel('Normalized Amplitude [dB]')
plt.show()