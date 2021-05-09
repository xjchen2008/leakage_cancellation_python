# G. T. Zhou and J. S. Kenney, "Predicting spectral regrowth of nonlinear power amplifiers," in IEEE Transactions on Communications, vol. 50, no. 5, pp. 718-722, May 2002, doi: 10.1109/TCOMM.2002.1006553.
import numpy as np
import matplotlib.pyplot as plt
#############
#Parameters
#############
c = 3e8
j = 1j
fs = 250e6#56*4e6  # Sampling freq
N = 4000  # This also limit the bandwidth. And this is determined by fpga LUT size.
T = N/fs  # T=N/fs#Chirp Duration
t = np.linspace(0, T, N)
f0 = 20e6 #-10e6  # -28e6 Start Freq
f1 = 40e6 #10e6  # 28e6 fs/2=1/2*N/T#End freq
K = (f1 - f0) / T # chirp rate = BW/Druation
phi0 = (4999+1) * np.pi / 10000  # Phase
f = np.linspace(0, fs-1, N)
freq = np.fft.fftfreq(N, d=1 / fs)
freq = np.linspace(0, fs, N)
distance = c * freq / K / 2.0
win=np.blackman(N)
##################
# Create the chirp
##################
x0 = 1*np.sin(2 * np.pi * (f0 * t + K / 2 * np.power(t, 2)))  # use this for chirp generation
xq0 = 1*np.sin(phi0 + 2 * np.pi * (f0 * t + K / 2 * np.power(t, 2)))  # use this for chirp generation
x = np.around(np.power(2,15)*np.multiply(x0, win))
xq = np.around(np.power(2,15)*np.multiply(xq0, win))
x_cx = x + j * xq
x_cx = np.multiply(x_cx, win)
#x_cx = np.multiply(x_cx,np.hamming(N))
#plt.figure(1)
#plt.plot(np.real(x_cx))


####################################################
d1 = 10 #9 # What happened to pulse compression if change the delay from 10 to 9. The phase discontinuity will smear the energy to other frequency. https://www.keysight.com/us/en/assets/7018-06760/application-notes/5952-8898.pdf?success=true
d2 = 1000
d3 = 300
y_cx_delay_1 = np.concatenate((x_cx[N-d1-1:-1], x_cx[:N-d1]), axis=0) # What happened to pulse compression if change the delay from 10 to 9. The phase discontinuity will smear the energy to other frequency.
y_cx_delay_2 = np.concatenate((x_cx[N-d2-1:-1], x_cx[:N-d2]), axis=0)
y_cx_delay_3 = np.concatenate((x_cx[N-d3-1:-1], x_cx[:N-d3]), axis=0)
vin1 = 1e-3*y_cx_delay_1 + 1e-7*y_cx_delay_2 + 1e-5*y_cx_delay_3 # With leakage, no intermodulatoin #1*y_cx_delay_1 + 1e-7*y_cx_delay_2 + 1e-5*y_cx_delay_3
vin2 = 1e-7*y_cx_delay_2 + 1e-5*y_cx_delay_3 # No leakage signal
vin2 = vin2 + 0.001*np.power(vin2,2)- 0.001*np.power(vin2,3) # Intermodulation of vin2
vin3 = vin1 + 0.001*np.power(vin1,2)- 0.001*np.power(vin1,3) # Intermodulation of vin1 with leakage

# vin2 = y_cx_received # With leakage signal

V0 = np.fft.fft(x_cx)#20*np.log10(np.fft.fft(x_cx))
V1 = np.fft.fft(vin1) #20*np.log10(np.fft.fft(vin1))
V2 = np.fft.fft(vin2)#20*np.log10(np.fft.fft(vin2))
V3 = np.fft.fft(vin3)#20*np.log10(np.fft.fft(vin3))
plt.plot(freq, V0-max(V0))
plt.figure()
plt.plot(freq/1e6, V1-max(V1),'b', freq/1e6, V2-max(V2),'k', freq/1e6, V3-max(V3),'r')
#plt.plot(freq/1e6, vin1,'b', freq/1e6, vin2,'k', freq/1e6, vin3,'r')
plt.title(' Intermodulated Chirp in Frequency Domain')
plt.xlabel('Frequency [MHz]')
plt.ylabel('Normalized Amplitude [dB]')
plt.grid()
plt.legend(['No Intermodulation with leakge signal',
           'Intermodulation without leakage signal',
           'Intermodulation with leakage signal'])
######################################
# Pulse Compression Stretching Method
#####################################
def PulseCompr(rx,tx,win):
    A = np.multiply(rx,win) # Add window here
    B = tx

    PC = np.fft.fft(A*np.conj(B))
    PC = 20 * np.log10(abs(PC))
    return PC

PC = PulseCompr(vin1,x_cx,np.blackman(N))
PC_intermod = PulseCompr(vin2,x_cx,np.blackman(N))
PC_intermod_leakage = PulseCompr(vin3,x_cx,np.blackman(N))
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
plt.figure()
#plt.plot(distance/1e3,PC-max(PC), '--', distance/1e3,PC_intermod-max(PC), 'k-', distance/1e3, PC_intermod_leakage-max(PC), 'r.-')
plt.plot(distance/1e3,PC-max(PC), 'b', distance/1e3,PC_intermod-max(PC), 'k', distance/1e3, PC_intermod_leakage-max(PC), 'r')
plt.grid()
plt.legend(['No Intermodulation with leakge signal',
           'Intermodulation without leakage signal',
           'Intermodulation with leakage signal'])

plt.title('Pulse Compression')
#plt.xlim((-10,500))
plt.xlabel('Distance km')
plt.ylabel('Normalized Power dB')
plt.show()