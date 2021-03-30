import numpy as np
import matplotlib.pyplot as plt
import readbin
import gradient_descent as gd

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

######################################
# Pulse Compression Stretching Method
#####################################


def PulseCompr(rx,tx,win):
    A = np.multiply(rx,win) # Add window here
    B = tx
    PC = 20*np.log10(abs(np.fft.fft(A*np.conj(B))))
    return np.squeeze(PC)

########################
# Gradient Decent
########################
# Template
psi_orth = tx_sig
psi_orth_delay_0 = np.concatenate((np.zeros(0), psi_orth[:N - 0]), axis=0)
psi_orth_delay_1 = np.concatenate((np.zeros(1), psi_orth[:N - 1]), axis=0)
psi_orth_delay_2 = np.concatenate((np.zeros(2), psi_orth[:N - 2]), axis=0)
psi_orth_delay_3 = np.concatenate((np.zeros(3), psi_orth[:N - 3]), axis=0)
psi_orth_delay_4 = np.concatenate((np.zeros(4), psi_orth[:N - 4]), axis=0)
psi_orth_delay_5 = np.concatenate((np.zeros(5), psi_orth[:N - 5]), axis=0)
psi_orth_delay_6 = np.concatenate((np.zeros(6), psi_orth[:N - 6]), axis=0)
psi_orth_delay_7 = np.concatenate((np.zeros(7), psi_orth[:N - 7]), axis=0)
psi_orth_delay_8 = np.concatenate((np.zeros(8), psi_orth[:N - 8]), axis=0)
psi_orth_delay_9 = np.concatenate((np.zeros(9), psi_orth[:N - 9]), axis=0)
H = np.matrix(np.transpose(
    [psi_orth_delay_0, psi_orth_delay_1, psi_orth_delay_2, psi_orth_delay_3, psi_orth_delay_4, psi_orth_delay_5,
     psi_orth_delay_6, psi_orth_delay_7, psi_orth_delay_8, psi_orth_delay_9]))


np.random.seed(100)
theta = np.random.randn(H.shape[1],1)
lr = 0.05
n_iter = 10000
#X = 2 * np.random.rand(N,1)
#y = 4 +3 * X+np.random.randn(N,1)
X = H#tx_sig#rx_avg#tx_sig
y = rx_avg#tx_sig#rx_avg#tx_sig#rx_avg-tx_sig
X = X[np.newaxis].T
y = y[np.newaxis].T
threshold = 1e-11 # not used?
theta,cost_history,theta_history = gd.gradient_descent(X,y,theta,lr,n_iter, threshold)

x_canc_gd = np.squeeze(np.array(np.dot(H,np.conj(theta)))) # as an array not matrix
print(theta)

fig,ax = plt.subplots(figsize=(12,8))
ax.set_ylabel('J(Theta)')
ax.set_xlabel('Iterations')
_=ax.plot(range(n_iter),10*np.log10(cost_history),'b.')

#################
# Plot
#################
plt.figure()
plt.plot(rx_avg)
plt.title('Received Signal in Time Domain')
plt.figure()
plt.plot(x_canc_gd)
plt.title('Cancellation Signal in Time Domain')
y_canc = rx_avg - x_canc_gd
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

plt.show()
