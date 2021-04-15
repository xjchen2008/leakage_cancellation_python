import coe_wavetable_4096_noise
import matplotlib.pyplot as plt
import numpy as np
t = coe_wavetable_4096_noise.t
N = coe_wavetable_4096_noise.N
freq = coe_wavetable_4096_noise.freq
v1 = 2*coe_wavetable_4096_noise.y_cx_0 # leakage
v2 = v1#0.01*coe_wavetable_4096_noise.y_cx_0_delay #np.zeros(N) #coe_wavetable_4096.y_cx_0_delay # Signal of Interest
vo = (v1+v2) + 0.001*np.power((v1+v2),2)- 0.001*np.power((v1+v2),3)#+ 0.1*np.power((v1+v2),4)#+ 0.1*np.power((v1+v2),5) #+ np.power((v1+v2),2) #+ 0.01*np.power((v1+v2),3) #+ np.power((v1+v2),4) # output voltage after nonlinear system such as LNA
Vo = 20*np.log10(np.fft.fft(vo))
V1 = 20*np.log10(np.fft.fft(v1))
############
# Pulse Compression
############
win=np.blackman(N)
A = np.conj(np.multiply(v1, win))
B = v2 # Frequency domain of reference signal. Conjugate of tx has already been counted in previous line.
C = vo
pc1 = 20*np.log10(np.abs(np.fft.fft((np.multiply(A, B)))))
pc2 = 20*np.log10(np.abs(np.fft.fft((np.multiply(A, C)))))
plt.figure()
plt.plot(freq, pc1, freq, pc2)

#############
# Plot
#############
#plt.figure()
#plt.plot(t, v1, t, v2)
plt.figure()
plt.plot(freq, V1-max(Vo), freq, Vo-max(Vo))
plt.title(' Intermodulated Chirp in Frequency Domain')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Normalized Amplitude [dB]')
plt.grid()
plt.show()

