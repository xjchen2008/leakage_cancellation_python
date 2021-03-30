import gradient_descent_E312 as gd
import timeit
import numpy as np
import time
import struct
import os
import matplotlib.pyplot as plt


def readbin1(filename, nsamples, fsize, rubish):
    bytespersample = 4
    samplesperpulse = nsamples
    total_samples = fsize / bytespersample
    total_pulse = total_samples / samplesperpulse
    file = open(filename, 'rb')
    file.seek((total_pulse - 1) * bytespersample * samplesperpulse)  # find the last pulse
    x = file.read(bytespersample * samplesperpulse)
    file.close()
    fmt = ('%sh' % (len(x) / 2))  # e.g.  '500h' means 500 shorts
    x_sig = np.array(struct.unpack(fmt, x)).astype(float)  # convert to complex float
    rx_sig = x_sig[1::2] + 1j * x_sig[0::2]
    rx_sig = rx_sig / 32767.0
    return rx_sig


def get_slice(data, pulselenth):
    data_ch0 = []
    data_ch1 = []
    for i in range(0, len(data), 2 * pulselenth):
        data_ch0[i:i + pulselenth] = data[i:i + pulselenth]
        data_ch1[i:i + pulselenth] = data[i + pulselenth:i + 2 * pulselenth]
    return data_ch0, data_ch1


def last_pulse(filename, nsamples, rubish):
    fsize = int(os.stat(filename).st_size)
    signal = readbin1(filename, nsamples, fsize, rubish)
    signal_ch0, signal_ch1 = get_slice(signal, 256)
    return signal_ch0, signal_ch1


j = 1j
fs = 28e6  # Sampling freq
N = 4096  # This also limit the bandwidth. And this is determined by fpga LUT size.
tc = N / fs  # T=N/fs#Chirp Duration
t = np.linspace(0, tc, N)
f0 = -14e6  # Start Freq
f1 = 14e6  # fs/2=1/2*N/T#End freq
K = (f1 - f0) / tc  # chirp rate = BW/Druation
# Chirp
#x0 = 1 * np.sin(2 * np.pi * (f0 * t + K / 2 * np.power(t, 2)))  # use this for chirp generation
#xq0 = 1 * np.sin(2 * np.pi * (f0 * t + K / 2 * np.power(t, 2)) - np.pi / 2)  # use this for chirp generation

# Square Wave
#x0 = np.concatenate((np.zeros(2048), np.ones(2048)))
#xq0 = np.concatenate((np.zeros(2048), np.ones(2048)))

# Sine wave
x0 = 1.0 * np.sin(2 * np.pi * fs / N * t)
xq0 = 1.0 * np.sin(2 * np.pi * fs / N * t - np.pi / 2)

X = x0 + j * xq0
X = np.reshape(X, [N, 1])

np.random.seed(100)
theta = np.random.randn(1, 1) + 0.1j  # parameter to learn


start = timeit.default_timer()
nsamples = N
channels = 2

for i in range(100):
    # Read latest pulse
    data_ch0, data_ch1 = last_pulse("usrp_samples_loopback.dat", nsamples * channels, 0)
    rx_error = np.array(data_ch0[0:N])
    # Gradient Decent
    theta, y_hat, cost_history = gd.gd(theta, rx_error, X)
    # save y_hat
    y_cxnew = np.around(32767 * y_hat)  # numpy.multiply(y_cx,win)
    yw = np.zeros(2 * N)

    for i in range(0, N):
        yw[2 * i + 1] = np.imag(y_cxnew[i])  # tx signal
        yw[2 * i] = np.real(y_cxnew[i])  # tx signal
    yw = np.int16(yw)  # E312 setting --type short
    filetime = str(time.gmtime().tm_sec)
    # data = open('usrp_samples4096_chirp_28MHz'+filetime+'.dat', 'w')
    data = open('usrp_samples4096_chirp_28MHz.dat', 'w')
    data.write(yw)
    data.close()


    plt.plot(y_hat)
plt.show()
stop = timeit.default_timer()
print('Time: ', stop - start)
