from readbin import readbin2
from matplotlib import pyplot as plt
import numpy as np

Nrx = 5625
PulseNum = 3
PulseIndex =  np.linspace(1,PulseNum, PulseNum)
sig = readbin2("20190320T095927.dat", np.short,PulseNum*Nrx)
x = np.linspace(0, len(sig)-1, len(sig))
plt.figure(1)
plt.plot(x, 32767*np.imag(sig),x, 32767*np.real(sig),'.-')
#plt.close()

phase = np.angle(sig) * 180/np.pi # phase in degree
phase_init = phase[10::Nrx] # Initial phase separated by Nrx points
print(np.angle(1j), phase_init, phase.dtype, len(phase), phase)

plt.figure(2)
plt.plot(PulseIndex, phase_init)
plt.xlabel("Pulse Number")
plt.ylabel("Initial Phase")
a = [1,2,3,4,5,6,7,8,9,0]
b = a[0::3]
print b

plt.grid()
plt.show()