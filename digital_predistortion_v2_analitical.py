#  "A robust Digital baseband predistorter constructed using memory polynomials" Lei Ding, Tong Zhou, Dennis R. Morgan, Zhengxiang Ma, J. Stevenson Kenney jaehyeong Kim, Charles R. Giardina.
import numpy as np
import timeit
import os
import struct
from numpy import fft
import skrf as rf
import time
import matplotlib.pyplot as plt
from scipy import signal
import UploadArb_CH2
import coe_wavetable_4096 as coe
import functions
import readosc
from functions import plot_freq_db as plot_freq

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
    A1 = X[0:int(N / 2), :]
    A2 = np.zeros([(upsamp_rate - 1) * N, D])
    A3 = X[int(N / 2):N, :]
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
    #win = np.blackman(N)
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
    M = int(N / 2)
    X1 = X[int(N / 2) - M: int(N / 2) + M, :]
    for i in range(3):
        X1 = zca_whitening_matrix(X1)
    np.save('tx_template_order9_delay1_upsamp100_28MHz_x1', X1)

    return X1


def save_tx_canc(filename, N, y_hat):
    # save y_hat
    # y_hat = np.concatenate((y_hat[N - 2028+2048:], y_hat[:N - 2028+2048]), axis=0) # fixed delay for tx chain from FPGA to RF out which need to measured for cancellation path
    y_hat = np.concatenate((y_hat[N:], y_hat[:N]),
                           axis=0)  # fixed delay for tx chain from FPGA to RF out which need to measured for cancellation path
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


def mem_poly_pd(Phi, w):
    y = np.dot(Phi, w)
    return y


def gd_pd(Phi_x, error, w):
    # gradient decent
    N = len(z)
    #Phi = Phi_x - Phi_y
    eta = 0.1  # 1  # learning rate
    w = w - (1.0 / N) * eta * np.dot(np.conjugate(Phi_x.T), error)
    return w


def cal_model_parameter(H, y):
    y = np.transpose(y[np.newaxis])
    print('Shape of y is', y.shape)
    # Calculate model parameter analytically.
    R_H = np.dot(np.matrix.getH(H),H) # Covariance Matrix of H
    print('Shape of R_H is', R_H.shape)
    R_yH = np.dot(np.matrix.getH(H),y)# Cross-covariance Matrix of {y,H}
    print('Shape of R_yH is', R_yH.shape)
    w = np.dot(np.linalg.inv(R_H),R_yH)
    return w


def PA(x):
    #X = np.fft.fft(x)
    h = 1#np.ones(N)# transfer function of PA
    #H = 1#np.fft.fft(h)
    #y =np.fft.ifft(np.multiply(X, H))
    #y = x + 0.001 * np.power(x, 2) - 0.1 * np.power(x, 3)
    #y = x + 0.001 * np.power(x, 2) - 0.1 * np.power(x, 3) # cannot be power!!!
    #y = x - 0.1 * np.power(abs(x), 2)*x # power(x, 3) is different from this!
    y = x - 0.1 * np.power(x, 3)  # cannot be power!!!
    return np.squeeze(y)


def phi_gen(x, order, delay):
    D = order*delay
    phi = 1j * np.ones([len(x), D])

    # x_cx_order = j * np.ones([N * upsamp_rate])
    #q0 = 0  # 180 # initial time delay for saving matrix space, take antenna cable into account
    #q = 0  # delay tap
    digital_filter_length = delay

    for q in range(delay):
        for k in range(order):
            idx = q * order + k
            power_order = (2*k)
            #x_cx_order = np.power(abs(x), power_order)* x
            x_cx_order = np.power(x, power_order) * x
            x_cx_delay = np.roll(x_cx_order, q)
            phi[:, idx] = x_cx_delay
            print ("digital_filter_length", digital_filter_length, "idx=", idx, "k=", k, "power order =", power_order, ', delay tap=', q)

    #for i in range(3):
    #    phi = zca_whitening_matrix(phi)
    return phi


if __name__ == "__main__":  # Change the following code into the c++
    N = 4000
    n = np.linspace(1,N,N)
    # initialization
    K = 50 # order
    Q = 1 # delay
    win = np.blackman(N)
    # Given conditions
    u = coe.y_cx#0.5*(coe.y_cx_sine+ coe.y_cx_sine2) #coe.y_cx_sine #0.5*(coe.y_cx_sine+ coe.y_cx_sine2)  # original ideal chirp. This is given.
    u = np.multiply(u, win)
    #y = PA(u)  # Output from PA before predistortion. This can be measured.
    y = readosc.readcsv(filename='data/test_sine_DPD_before.csv')
    y = signal.hilbert((y/max(abs(y))))
    U = np.fft.fft(u)
    U_log = 20 * np.log10(abs(U))
    Y = np.fft.fft(y, axis = 0)
    Y_log = 20 * np.log10(abs(Y))
    # Calculation
    # forward
    Phi_u = phi_gen(u, order=K, delay=Q)
    b = cal_model_parameter(Phi_u, y)
    y_hat = np.dot(Phi_u, b)
    Y_hat = np.fft.fft(y_hat, axis = 0)
    Y_hat_log = 20 * np.log10(abs(Y_hat))
    plot_freq(coe.freq/1e6, y,'b')
    plot_freq(coe.freq/1e6, y_hat,'r')
    plt.legend(['y','y_hat'])
    # inverse
    Phi_y = phi_gen(y, order=K, delay=Q)
    b_inv = cal_model_parameter(Phi_y, u)
    u_hat = np.dot(Phi_y, b_inv)
    plt.figure()
    plot_freq(coe.freq, u,'b')
    plot_freq(coe.freq, u_hat,'r')
    plt.legend(['u','u_hat'])

    # DPD
    x = np.array(np.dot(Phi_u, b_inv))
    x = x.squeeze()
    x = x / abs(x.max)
    UploadArb_CH2.UploadArb(x.real)
    readosc.readosc(filename='data/test_sine_DPD_after.csv')
    y_DPD = readosc.readcsv(filename='data/test_sine_DPD_after.csv')
    #y_DPD = PA(x)  # DPD output
    plt.figure()


    plot_freq(coe.freq/1e6, u,'b')
    plot_freq(coe.freq/1e6, y,'r')
    plot_freq(coe.freq/1e6, y_DPD,'k')
    plt.legend(['Input', 'y','y_DPD'])
    #plt.figure()
    #plt.plot(y_DPD)
    #plt.plot(u)

    # Pulse Compression
    y_DPD = np.multiply(y_DPD, win)
    pc_DPD = functions.PulseCompr(tx = u, rx = y_DPD, win = win)
    pc_DPD_normalize = pc_DPD - max(pc_DPD)


    y= np.multiply(y, win)
    pc = functions.PulseCompr(tx=u, rx=y, win = win)
    pc_normalize = pc - max(pc)
    plt.figure()
    plt.plot(coe.distance, pc_normalize,'b')
    plt.plot(coe.distance, pc_DPD_normalize,'r')
    plt.legend(['Without DPD', 'With DPD'])
    plt.title('Pulse Compression')

    #plt.plot(coe.freq, U_hat_log)
    #plt.plot(coe.freq, X_log)
    #plt.plot(coe.freq, Y_log)
    #plt.plot(coe.freq, U_check_log)
    plt.show()