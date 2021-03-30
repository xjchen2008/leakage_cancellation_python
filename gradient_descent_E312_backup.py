import numpy as np
import timeit
import os
import struct
import time
#import matplotlib.pyplot as plt

def cal_cost(rx_error):
    m = len(rx_error)
    A = rx_error.reshape([m,1]) # y_prediction - y
    cost = 1.0 / (2 * m) * np.dot(np.conj(A).T, A)[0][0]
    return cost


def readbin1(filename, nsamples, fsize, rubish):
    bytespersample = 4
    samplesperpulse = nsamples
    total_samples = fsize/bytespersample
    total_pulse = total_samples / samplesperpulse
    file = open(filename,'rb')
    file.seek((total_pulse-1) * bytespersample*samplesperpulse+ bytespersample*rubish) # find the last pulse
    x = file.read( bytespersample*samplesperpulse)
    file.close()
    fmt = ('%sh' % (len(x) /2))  # e.g.  '500h' means 500 shorts
    x_sig = np.array(struct.unpack(fmt, x)).astype(float) # convert to complex float
    rx_sig = x_sig[1::2] + 1j*x_sig[0::2]
    rx_sig = rx_sig / 32767.0
    return rx_sig


def get_slice(data, pulselenth):
    data_ch0 = []
    data_ch1 = []
    for i in range(0,len(data), 2*pulselenth):
        data_ch0[i:i+pulselenth] = data[i:i+pulselenth]
        data_ch1[i:i+pulselenth] = data[i+pulselenth:i + 2*pulselenth]
    return data_ch0, data_ch1


def last_pulse (filename, nsamples, rubish):
    fsize = int(os.stat(filename).st_size)
    signal = readbin1(filename, nsamples, fsize, rubish)
    signal_ch0, signal_ch1 = get_slice(signal, 256)
    return signal_ch0, signal_ch1


def gd(theta, rx_error_sim, X):
    # gradient decent
    eta = 1  # learning rate
    m = len(rx_error_sim)
    c2 =0.43 # this is a amplitude calibration coefficient = Rx/Tx
    #theta = theta -(1.0 / m) * eta * np.dot(np.conj(X.T), rx_error_sim)
    theta = theta + (1.0 / m) * eta * np.dot(np.real(X.T), np.real(rx_error_sim)/ c2)
    y_hat = np.dot((X), theta) # y_hat = prediction for cancellation signal
    cost_history = 10*np.log10(cal_cost(rx_error_sim))
    return theta, y_hat, cost_history

def tx_template(N):
    j = 1j
    fs = 28e6  # Sampling freq
    tc = N / fs  # T=N/fs#Chirp Duration
    t = np.linspace(0, tc, N)
    f0 = -14e6  # Start Freq
    f1 = 14e6  # fs/2=1/2*N/T#End freq
    K = (f1 - f0) / tc  # chirp rate = BW/Druation
    win = np.blackman(N)
    # win=np.hamming(N)
    win=1
    # Sine wave
    x0 = -1.0 * np.sin(2 * np.pi * fs / N * t)
    xq0 = -1.0 * np.sin(2 * np.pi * fs / N * t - np.pi / 2)

    # Chirp
    #x0 = 1 * np.sin(2 * np.pi * (f0 * t + K / 2 * np.power(t, 2)))  # use this for chirp generation
    #xq0 = 1 * np.sin(2 * np.pi * (f0 * t + K / 2 * np.power(t, 2)) - np.pi / 2)  # use this for chirp generation
    # Square Wave
    #x0 = np.concatenate((np.zeros(2048),np.ones(2048)))
    #xq0 = np.concatenate((np.zeros(2048),np.ones(2048)))
    X = x0 + j * xq0
    X = np.multiply(X, win)
    #X_cx_delay = np.concatenate((X[N - 354:], X[:N - 354]), axis=0)
    #X = X_cx_delay
    X = np.reshape(X, [N, 1])
    return X


def save_tx_canc(filename, N,y_hat):
    # save y_hat
    y_hat = np.concatenate((y_hat[N - 2028+2048:], y_hat[:N - 2028+2048]), axis=0) # fixed delay for tx chain from FPGA to RF out which need to measured for cancellation path
    y_cxnew = np.around(32767 * y_hat )  # numpy.multiply(y_cx,win) 6.9
    #y_cxnew = np.around(32767 * y_hat/max(y_hat))  # numpy.multiply(y_cx,win) 6.9

    yw = np.zeros(2 * N)

    for i in range(0, N):
        yw[2 * i + 1] = np.imag(y_cxnew[i])  # tx signal
        yw[2 * i] = np.real(y_cxnew[i])  # tx signal
    yw = np.append(yw, yw[-2]) # Manually add one more point at the end of the 4096 points pulse to match the E312 setup
    yw = np.append(yw, yw[-2])
    yw = np.int16(yw)  # E312 setting --type short

    # filetime = str(time.gmtime().tm_sec)
    # data = open('usrp_samples4097_chirp_28MHz'+filetime+'.dat', 'w')
    data = open(filename, 'w')
    data.write(yw)
    data.close()


def rx_sim(N):
    j = 1j
    fs = 28e6  # Sampling freq
    tc = N / fs  # T=N/fs#Chirp Duration
    t = np.linspace(0, tc, N)
    f0 = -14e6  # Start Freq
    f1 = 14e6  # fs/2=1/2*N/T#End freq
    K = (f1 - f0) / tc  # chirp rate = BW/Druation
    #win = np.blackman(N)
    # win=np.hamming(N)
    win=1
    Amp = 0.5
    phi_init1 = 0#-105.0 / 180.0 * np.pi
    phi_init2 = 0#-105.0 / 180.0 * np.pi
    dc_offset = 0

    # Sine wave
    y0 = 1.0 * np.sin(2 * np.pi * fs / N * t)
    yq0 = 1.0 * np.sin(2 * np.pi * fs / N * t - np.pi / 2)

    #y0 = Amp * np.sin(2 * np.pi * fs / N * t + phi_init1) + dc_offset # + numpy.sin(4*numpy.pi*fs/N*t)# just use LO to generate a LO. The
    #yq0 = Amp * np.sin(2 * np.pi * fs / N * t - np.pi / 2 + phi_init2) + dc_offset  # + numpy.sin(4*numpy.pi*fs/N*t-numpy.pi/2)

    # Chirp
    #y0 = Amp * np.sin(phi_init1 + 2 * np.pi * (f0 * t + K / 2 * np.power(t, 2))) + dc_offset  # use this for chirp generation
    #yq0 = Amp * np.sin(phi_init2 + 2 * np.pi * (f0 * t + K / 2 * np.power(t, 2)) - np.pi / 2) + dc_offset  # use this for chirp generation
    # Square
    #y0 = np.concatenate((np.zeros(2048),np.ones(2048)))
    #yq0 = np.concatenate((np.zeros(2048),np.ones(2048)))
    y = y0 + j * yq0
    y = np.multiply(y, win)
    y_cx_delay = Amp * np.concatenate((y[N:], y[:N]), axis=0)

    y = np.reshape(y_cx_delay, [N, 1])
    return y


def read_last_pulse(filename, N):
    nsamples = N
    channels = 2
    rubish = 0
    # Read latest pulse
    data_ch0, data_ch1 = last_pulse(filename, nsamples * channels, rubish)
    rx_error = np.array(data_ch0[0:N])
    return rx_error


def main(theta, N=4096, rx_error_sim = np.zeros([4096, 1])):
    start = timeit.default_timer()
    X = tx_template(N)

    # Gradient Decent
    rx_error = read_last_pulse("usrp_samples_loopback.dat",N)

    theta_cx_in = theta[0] + 1j*theta[1] #only real part
    #theta_cx_out, y_hat, cost_history = gd(theta_cx_in, rx_error, X)
    theta_cx_out, y_hat, cost_history = gd(theta_cx_in, rx_error_sim, X) #uncomment this line out when using simulated received signal

    save_tx_canc('usrp_samples4096_chirp_28MHz_fixed_delayed.dat', N, y_hat) # add FGPA tx delay inside this function
    stop = timeit.default_timer()
    print('Time: ', stop - start)
    print('cost_history:', cost_history)
    print('Theta_cx_out:', theta_cx_out)
    theta_real_out = float(np.real(theta_cx_out))
    theta_imag_out = float(np.imag(theta_cx_out))
    #return theta_real_out, theta_imag_out
    return theta_real_out, theta_imag_out, y_hat #uncomment this line out when using simulated received signal

if __name__ == "__main__": # Change the following code into the c++
    np.random.seed(0)
    N = 4096   # This also limit the bandwidth. And this is determined by fpga LUT size.
    theta = 0.1#+0.1j# *np.random.randn(1, 1) + 0.1j  # parameter to learn
    theta_real = np.real(theta)
    theta_imag = np.imag(theta)
    theta_tuple = (theta_real, theta_imag)
    y = rx_sim(N)
    y_hat = np.zeros([N, 1])
    start = timeit.default_timer()

    for itt in range(30):
        rx_error_sim = y_hat + y  # uncomment this line out when using simulated received signal
        theta_real, theta_imag, y_hat = main( theta_tuple, N, rx_error_sim) # uncomment this line out when using simulated received signal
        theta_tuple = (theta_real, theta_imag) # uncomment this line out when using simulated received signal


        #theta_real, theta_imag = main(theta_tuple, N)
        print('this is not the main function but the top level')
        print(itt)
    stop = timeit.default_timer()
    print('Time: ', stop - start)


