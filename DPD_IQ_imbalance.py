import numpy as np
import timeit
import os
import struct
from numpy import fft
#import skrf as rf
import time
import matplotlib.pyplot as plt
from gradient_descent_E312 import read_last_pulse


def sample_cov(X):
    # Check the math here https://www.itl.nist.gov/div898/handbook/pmc/section5/pmc541.htm
    # Estimation of covariance matrices: https://en.wikipedia.org/wiki/Covariance_matrix
    X = np.matrix(X)
    N = X.shape[0]  # rows of X: number of observations
    D = X.shape[1]  # columns of X: number of variables
    mean_col = 1j * np.ones(D)  # has to define a complex number for keeping the imaginary part
    for col_indx in range(D):
        mean_col[col_indx] = np.mean(X[:, col_indx])
    Mx = X - mean_col  # Zero mean matrix of X
    S = np.dot(Mx.H, Mx) / (N - 1)  # sample covariance matrix
    return np.conj(
        S), Mx  # add a np.conj() because when I compare to the numpy.cov() the result only be the same when adding the conjucate... strange.


def zca_whitening_matrix(X0):
    """
    Function to compute ZCA whitening matrix (aka Mahalanobis whitening).
    INPUT:  X0: [N x D] matrix.
        Rows: Observations
        Columns: Variables
    ZCAMatrix: [D x D] transformation matrix
    OUTPUT: Y = (X0 -X_mean)W. Its covariance matrix is identity matrix
    """
    N = X0.shape[0]
    # Sample Covariance matrix [column-wise variables]: Sigma = (X-mu)' * (X-mu) / (N-1)
    sigma0 = np.cov(X0, rowvar=False)  # [D x D]
    # print(sigma0)
    sigma, Mx = sample_cov(X0)
    # print(sigma)

    XhX = np.dot(Mx.H, Mx)  # (N-1)*sigma should be the same but there is a conjugate difference. Don't know why...
    # Singular Value Decomposition. X = U * np.diag(S) * V
    U, S, Vh = np.linalg.svd(XhX)
    # U: [D x D] eigenvectors of sigma.
    # S: [D x 1] eigenvalues of sigma.
    # V: [D x D] transpose of U
    # Whitening constant: prevents division by zero
    epsilon = 1e-1000
    ZCAMatrix = np.sqrt(N - 1) * np.dot(Vh.H, np.dot(np.diag(1.0 / np.sqrt(S + epsilon)), Vh))  # [M x M]

    Y = np.dot(Mx, ZCAMatrix)
    cov_Y, Mx = sample_cov(Y)
    #print(np.diag(cov_Y))  # Every time call this func will print. Should be all 1. It means basis are independent.

    # plt.matshow(abs(cov_Y))
    # plt.show()
    return Y


def array2tuple(array, D):
    theta_tuple = ()
    for i in range(D):
        arr = [(np.real(array[i]), np.imag(array[i]))]
        theta_tuple = theta_tuple + tuple(map(tuple, arr))
    return theta_tuple


def cal_cost(rx_error):
    m = len(rx_error)
    A = rx_error.reshape([m, 1])  # y_prediction - y
    cost = 1.0 / (2 * m) * np.dot(np.conj(A).T, A)[0][0]
    return cost


def upsampling(x, upsamp_rate):
    # Actually no need. Just use higher fs to generate better template digitally is good enough.
    # This is just a one-dimensional interpolation.
    # https://dsp.stackexchange.com/questions/14919/upsample-data-using-ffts-how-is-this-exactly-done
    # FFT upsampling method

    N = x.shape[0]
    D = x.shape[1]
    # To frequency domain
    X = fft.fft(x, axis=0)
    # Add taps in the middle
    A1 = X[0:N / 2, :]
    A2 = np.zeros([(upsamp_rate - 1) * N, D])
    A3 = X[N / 2:N, :]
    XX = np.concatenate((A1, A2, A3))
    # To time domain
    xx = upsamp_rate * fft.ifft(XX, axis=0)
    # plt.plot(np.linspace(0,1, xx.shape[0]), xx)
    # plt.plot(np.linspace(0,1, xx.shape[0]), xx ,'ko')
    # plt.plot(np.linspace(0,1, N), x, 'y.')
    # plt.show()
    x_upsamp = np.reshape(xx, (N * upsamp_rate, D))  # change back to 1-D
    return x_upsamp


def downsampling(x, downsamp_rate):
    N = x.shape[0]
    D = x.shape[1]
    x_downsamp = x[::downsamp_rate]
    # plt.plot(np.linspace(0,1, x_downsamp.shape[0]), x_downsamp)
    # plt.plot(np.linspace(0,1, x_downsamp.shape[0]), x_downsamp ,'ko')
    # plt.plot(np.linspace(0,1, N), x, 'y.')
    # plt.show()
    return x_downsamp


def rx_sim(N, upsamp_rate, delay_tap):
    j = 1j
    fs = 28e6  # Sampling freq
    tc = N / fs  # T=N/fs#Chirp Duration
    t = np.linspace(0, tc, N)
    f0 = -10e6  # Start Freq
    f1 = 10e6  # fs/2=1/2*N/T#End freq
    K = (f1 - f0) / tc  # chirp rate = BW/Druation
    # win = np.blackman(N)
    win=np.hamming(N)
    win = 1
    Amp0 = 0.1
    Amp1 = 0.10
    phi_init1 = 1*5.0 / 180 * np.pi  # -105.0 / 180.0 * np.pi
    phi_init2 = 0*45.0 / 180 * np.pi # -105.0 / 180.0 * np.pi
    dc_offset0 = 0
    dc_offset1 = 0.01

    # Sine wave
    # y0 = Amp * np.sin(2 * np.pi * fs / N * t)
    # yq0 = Amp * np.sin(2 * np.pi * fs / N * t - np.pi / 2)

    # y0 = Amp * np.sin(2 * np.pi * fs / N * t + phi_init1) + dc_offset # + numpy.sin(4*numpy.pi*fs/N*t)# just use LO to generate a LO. The
    # yq0 = Amp * np.sin(2 * np.pi * fs / N * t - np.pi / 2 + phi_init2) + dc_offset  # + numpy.sin(4*numpy.pi*fs/N*t-numpy.pi/2)

    # Chirp
    y0 = Amp0 * np.sin(
        phi_init1 + 2 * np.pi * (f0 * t + K / 2 * np.power(t, 2))) + dc_offset0  # use this for chirp generation
    yq0 = Amp1 * np.sin(phi_init2 + 2 * np.pi * (f0 * t + K / 2 * np.power(t, 2)) - np.pi / 2) + dc_offset1  # use this for chirp generation

    # Square
    # y0 = np.concatenate((np.zeros(2048),np.ones(2048)))
    # yq0 = np.concatenate((np.zeros(2048),np.ones(2048)))
    y0[3000] = .3 # simulate the imperfect spike in the tx template signal
    y0[1000] = .3 # simulate the imperfect spike in the tx template signal
    y = y0 + j * np.roll(yq0,0)
    y = np.multiply(y, win)
    # Using filter measured frequency response with group delay
    #s = touchstone('LBAND_FILTER_TOUCHSTONE_20MHZ_4096.S2P')
    #s21 = s[:, 1, 0]
    #y_gd = fft.ifft(fft.fft(y * s21))

    # y_cx_delay = np.concatenate((y[N-0:], y[:N-0]), axis=0)
    y_upsamp = upsampling(np.reshape(y, (N, 1)), upsamp_rate)  # step 1: up-sampling
    y_upsamp = np.reshape(y_upsamp, N * upsamp_rate)
    y_delay = np.roll(y_upsamp, delay_tap)  # shift 1 tap
    y = downsampling(np.reshape(y_delay, (N * upsamp_rate, 1)), upsamp_rate)

    y = np.reshape(y, [N, ])
    return y


def tx_template(N, D, upsamp_rate):
    j = 1j
    # fs = 28e6*upsamp_rate  # Sampling freq
    # N = upsamp_rate * N
    fs = 28e6  # Sampling freq
    # N = Norder_rep
    tc = N / fs  # T=N/fs#Chirp Duration
    t = np.linspace(0, tc, N)
    f0 = -10e6  # Start Freq
    f1 = 10e6  # fs/2=1/2*N/T#End freq
    K = (f1 - f0) / tc  # chirp rate = BW/Druation
    phi_init = 0
    # win = np.blackman(N)
    # win=np.hamming(N)
    win = 1
    # Sine wave
    # x0 = 1 * np.sin(2 * np.pi * 1.0 * fs / N * t + phi_init)  # + numpy.sin(4*numpy.pi*fs/N*t)# just use LO to generate a LO. The
    # xq0 = 1 * np.sin(2 * np.pi * 1.0 * fs / N * t - np.pi / 2 + phi_init)  # + numpy.sin(4*numpy.pi*fs/N*t-numpy.pi/2)
    # Chirp
    x0 = 1 * np.exp(j * 0) * np.sin(2 * np.pi * (f0 * t + K / 2 * np.power(t, 2)))  # use this for chirp generation
    xq0 = 1 * np.exp(j * 0) * np.sin(
        2 * np.pi * (f0 * t + K / 2 * np.power(t, 2)) - np.pi / 2)  # use this for chirp generation
    # Square Wave
    # x0 = np.concatenate((np.zeros(2048),np.ones(2048)))
    # xq0 = np.concatenate((np.zeros(2048),np.ones(2048)))

    x_cx = x0 + j * xq0
    x_cx = np.multiply(x_cx, win)  # add window


    x_upsamp = upsampling(np.reshape(x_cx, (N, 1)), upsamp_rate)  # step 1: up-sampling
    x_upsamp = np.reshape(x_upsamp, N * upsamp_rate)

    x_cx_delay = j * np.ones([N * upsamp_rate, D])
    # x_cx_order = j * np.ones([N * upsamp_rate])
    k0 = 0  # 180 # initial time delay for saving matrix space, take antenna cable into account
    k = 0  # delay tap
    order_idx = 1
    order = 1
    digital_filter_length = 300
    for i in range(D):
        # x_cx_delay[:,i] = np.roll(x_upsamp, 0+1*i) # here  #x_cx_delay[:,i] = np.roll(x_cx, 20*i) # here

        #print (
        #    "digital_filter_length", digital_filter_length, "order_idx=", order_idx, "order=", order,
        #    'delay tap,k=', k)
        x_cx_order = np.power(abs(x_upsamp), order - 1) * x_upsamp
        x_cx_delay[:, i] = np.roll(x_cx_order, k + k0)
        if k == digital_filter_length - 1:
            #print k
            k = 0
            order_idx += 1
            order = 2 * order_idx - 1
        else:
            k += 1

    x_cx_order = downsampling(x_cx_delay, upsamp_rate)
    X = np.matrix(x_cx_order)
    '''
    order = (5+1)/2
    k=0
    for i in range(D):
        # x_cx_delay[:,i] = np.roll(x_upsamp, 0+1*i) # here  #x_cx_delay[:,i] = np.roll(x_cx, 20*i) # here

        order_rep = i% order+1
        order_pow = 2 * order_rep - 1
        print ("order=",order,"order_rep=", order_rep,"order_pow=", order_pow,'k=',k)
        x_cx_order = np.power(abs(x_upsamp), order_pow) * x_upsamp
        x_cx_delay[:, i] = np.roll(x_cx_order, k)
        if order_rep%order_pow ==0 and i >= 2:#order:
            print k
            k += 1

    x_cx_delay = downsampling(x_cx_delay, upsamp_rate)
    X = np.matrix(x_cx_delay)
    '''
    # for i in range(3):
    #    X = zca_whitening_matrix(X)

    # np.save('tx_template_order9_delay1_upsamp100_28MHz', X)

    # Generate X for accelerating the gradient descent.
    M = N / 2
    X1 = X[N / 2 - M: N / 2 + M, :]
    #for i in range(3):
    #   X1 = zca_whitening_matrix(X1)
    np.save('tx_template_order9_delay1_upsamp100_28MHz_x1', X1)

    return X1


def save_tx_canc(filename, N, y_hat):
    # save y_hat
    # y_hat = np.concatenate((y_hat[N - 2028+2048:], y_hat[:N - 2028+2048]), axis=0) # fixed delay for tx chain from FPGA to RF out which need to measured for cancellation path
    # y_hat = np.concatenate((y_hat[N:], y_hat[:N]), axis=0)  # fixed delay for tx chain from FPGA to RF out which need to measured for cancellation path
    y_cxnew = np.around(32767 * y_hat)  # numpy.multiply(y_cx,win) 6.9

    yw = np.zeros(2 * N)

    for i in range(0, N):
        yw[2 * i + 1] = np.imag(y_cxnew[i])  # tx signal
        yw[2 * i] = np.real(y_cxnew[i])  # tx signal
    yw = np.append(yw,
                   yw[-2])  # Manually add one more point at the end of the 4096 points pulse to match the E312 setup
    yw = np.append(yw, yw[-2])
    yw = np.int16(yw)  # E312 setting --type short

    # filetime = str(time.gmtime().tm_sec)
    # data = open('usrp_samples4097_chirp_28MHz'+filetime+'.dat', 'w')
    data = open(filename, 'w')
    data.write(yw)
    data.close()


def plot(N, y, yc, x_tx):
     n = np.linspace(1, N, N)
     '''
     plt.figure()
     plt.plot(y.real,y.imag)
     plt.title('imbalance constellation')
     plt.figure()
     plt.plot(yc.real,yc.imag)
     plt.title('corrected constellation')
     '''

     plt.figure()
     plt.plot(n, y.real, n, y.imag)
     plt.title('IQ-imbalance and delay')
     plt.figure()

     plt.plot(n, yc.real, n, yc.imag)
     plt.title('corrected signal')
     plt.figure()
     plt.plot(n, x_tx.real, n, x_tx.imag)
     plt.title('tx signal')
     plt.show()


def main():
    start = timeit.default_timer()
    # initialization
    N = 4096
    x = np.squeeze(np.array(tx_template(N, 1, 1)))  # set input signal
    #y = rx_sim(N, 1, 0)
    y = np.zeros(N)
    # averaging the last pulses
    Navg = 2
    for idx in range(1, Navg):
        y0 = np.squeeze(np.reshape(read_last_pulse("usrp_samples_loopback.dat", N, idx), [N, 1]), axis= 1)
        y = y+y0
    y = y/(1.0*Navg)
    y = 15*y  # tunable digital scaling for setting the generated x_tx less than 1
    ########################
    X = np.fft.fft(x)
    Y = np.fft.fft(y)
    H = np.divide(Y, X)
    H_inv = 1/H

    print(np.fft.ifft(H_inv).real.mean())
    print(np.fft.ifft(H_inv).imag.mean())
    X_tx = np.multiply(H_inv, X)
    X_tx[0] = 0 # setting the bin zero(DC component to zero). It is not zero due to noise and randomness.
    x_tx = np.fft.ifft(X_tx)
    x_tx_normalize = x_tx / max(abs(x_tx))
    X_tx_normalize = np.fft.fft(x_tx_normalize)
    save_tx_canc('usrp_samples4096_chirp_28MHz_fixed_delayed_updated.dat', N, np.array(x_tx_normalize))  # add FGPA tx delay inside this function
    Yc = np.multiply(H, X_tx_normalize)
    yc = np.fft.ifft(Yc)
    plot(N, y, yc, x_tx_normalize) # uncomment this line for plotting

    stop = timeit.default_timer()
    print('Time: ', stop - start)


if __name__ == "__main__":  # Change the following code into the c++
    main()