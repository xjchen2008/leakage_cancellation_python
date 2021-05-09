import numpy as np
import timeit
import os
import struct
from numpy import fft
import skrf as rf
import time
import matplotlib.pyplot as plt
import readosc
import UploadArb
import coe_wavetable_4096
import setup
import functions
from scipy import signal


def touchstone(filename):
    N = 4096
    x = rf.Network(filename)
    x.frequency.unit = 'ghz'
    # x.interpolate_self_npoints(4096)
    f_new = rf.Frequency(1.4135 - 0.01, 1.4135 + 0.01, N)
    x_interp = x.interpolate(f_new, kind='cubic')
    gd = x_interp.s21.group_delay * 1e9  # in ns

    # plot group delay
    # x_interp.plot(gd)
    # plt.ylabel('Group Delay (ns)')
    # plt.title('Group Delay of Ring Slot S21')

    return x_interp.s


def sample_cov(X):
    # Check the math here https://www.itl.nist.gov/div898/handbook/pmc/section5/pmc541.htm
    # Estimation of covariance matrices: https://en.wikipedia.org/wiki/Covariance_matrix
    X = np.matrix(X)
    N = X.shape[0]  # rows of X: number of observations
    D = X.shape[1]  # columns of X: number of variables
    mean_col = np.ones(D)# 1j * np.ones(D)  # has to define a complex number for keeping the imaginary part
    for col_indx in range(D):
        mean_col[col_indx] = np.mean(X[:, col_indx])
    Mx = X - mean_col  # Zero mean matrix of X
    S = np.dot(Mx.H, Mx) / (N - 1)  # sample covariance matrix
    return np.conj(
        S), Mx  # add a np.conj() because when I compare to the numpy.cov() the result only be the same when adding the conjucate... strange.


def array2tuple(array, D):
    theta_tuple = ()
    for i in range(D):
        arr = [(np.real(array[i]), np.imag(array[i]))]
        theta_tuple = theta_tuple + tuple(map(tuple, arr))
    return theta_tuple


def cal_cost(X, rx_error):
    m = len(rx_error)
    A = rx_error.reshape([m, 1])  # y_prediction - y
    cost = 1.0 / (2 * m) * np.dot(np.conj(X).T, A)[0][0]
    cost2 = 1.0 / (2 * m) * np.dot(np.conj(A).T, A)[0][0] # this is a direct calculation of residual power
    return cost


def readbin1(filename, nsamples, fsize, rubish):
    bytespersample = 4
    samplesperpulse = nsamples
    total_samples = fsize / bytespersample
    total_pulse = total_samples / samplesperpulse
    file = open(filename, 'rb')
    file.seek(int(total_pulse - 1) * bytespersample * samplesperpulse + bytespersample * rubish)  # find the last pulse
    x = file.read(bytespersample * samplesperpulse)
    file.close()
    fmt = ('%sh' % (len(x) / 2))  # e.g.  '500h' means 500 shorts
    x_sig = np.array(struct.unpack(fmt, x)).astype(float)  # convert to complex float
    rx_sig = -x_sig[0::2] + 1j * x_sig[
                                 1::2]  # Important! Here I added a negtive sign here for calibrate the tx rx chain. There is a flip of sign somewhere but I cannot find.
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


def gd(theta, rx_error_sim, X1):
    # gradient decent
    N = len(rx_error_sim)
    M = int(len(X1) / 2)  # N/2
    eta = 0.1  # 1  # learning rate
    m = M / 2  # len(rx_error_sim)
    c2 = 1  # 0.45 # this is a amplitude calibration coefficient = Rx/Tx, this will also make the convergence faster.
    # X1 = X[N/2-M: N/2+M, :]
    # for i in range(3):
    #    X1 = zca_whitening_matrix(X1)
    # theta = theta - (1.0 / m) * eta * np.dot(np.conj(X[N/2-M: N/2+M,:].T), rx_error_sim[N/2-M: N/2+M] / c2)
    #theta = theta - (1.0 / m) * eta * np.dot(np.conj(X1.T),
    #                                         rx_error_sim[int(N / 2) - M: int(N / 2) + M] / c2)
    #theta = theta - (1.0 / m) * eta * np.dot(np.conj(X1.T), rx_error_sim)
    theta = theta - (1.0 / m) * eta * np.dot(X1.H, rx_error_sim)
    # theta = theta - (1.0 / m) * eta * np.dot(np.real(X.T), np.real(rx_error_sim)/ c2)
    y_hat = np.dot((X1), theta)  # y_hat = prediction for cancellation signal
    cost_history = 10 * np.log10(abs(cal_cost(X1, rx_error_sim)))
    return theta, y_hat, cost_history


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


def rx_sim(N, upsamp_rate, delay_tap):
    j = 1j
    fs = 28e6  # Sampling freq
    tc = N / fs  # T=N/fs#Chirp Duration
    t = np.linspace(0, tc, N)
    f0 = -10e6  # Start Freq
    f1 = 10e6  # fs/2=1/2*N/T#End freq
    K = (f1 - f0) / tc  # chirp rate = BW/Druation
    # win = np.blackman(N)
    # win=np.hamming(N)
    win = 1
    Amp = 1
    phi_init1 = 0 / 180  # -105.0 / 180.0 * np.pi
    phi_init2 = 0 / 180  # -105.0 / 180.0 * np.pi
    dc_offset = 0

    # Sine wave
    # y0 = Amp * np.sin(2 * np.pi * fs / N * t)
    # yq0 = Amp * np.sin(2 * np.pi * fs / N * t - np.pi / 2)

    # y0 = Amp * np.sin(2 * np.pi * fs / N * t + phi_init1) + dc_offset # + numpy.sin(4*numpy.pi*fs/N*t)# just use LO to generate a LO. The
    # yq0 = Amp * np.sin(2 * np.pi * fs / N * t - np.pi / 2 + phi_init2) + dc_offset  # + numpy.sin(4*numpy.pi*fs/N*t-numpy.pi/2)

    # Chirp
    y0 = Amp * np.sin(
        phi_init1 + 2 * np.pi * (f0 * t + K / 2 * np.power(t, 2))) + dc_offset  # use this for chirp generation
    yq0 = Amp * np.sin(phi_init2 + 2 * np.pi * (
            f0 * t + K / 2 * np.power(t, 2)) - np.pi / 2) + dc_offset  # use this for chirp generation

    # Square
    # y0 = np.concatenate((np.zeros(2048),np.ones(2048)))
    # yq0 = np.concatenate((np.zeros(2048),np.ones(2048)))
    y = y0 + j * yq0
    y = np.multiply(y, win)
    # Using filter measured frequency response with group delay
    s = touchstone('LBAND_FILTER_TOUCHSTONE_20MHZ_4096.S2P')
    s21 = s[:, 1, 0]
    y_gd = fft.ifft(fft.fft(y * s21))

    # y_cx_delay = np.concatenate((y[N-0:], y[:N-0]), axis=0)
    y_upsamp = upsampling(np.reshape(y, (N, 1)), upsamp_rate)  # step 1: up-sampling
    y_upsamp = np.reshape(y_upsamp, N * upsamp_rate)
    y_delay = np.roll(y_upsamp, delay_tap)  # shift 1 tap
    y = downsampling(np.reshape(y_delay, (N * upsamp_rate, 1)), upsamp_rate)

    y = np.reshape(y, [N, 1])
    return y


def read_last_pulse(filename, N, pulse_idx_from_back =0):
    nsamples = N
    channels = 2
    # Read latest pulse
    data_ch0, data_ch1 = last_pulse(filename, nsamples * channels, -2*(pulse_idx_from_back+1)*N) #two channels; pulse_idx_from_back+1 because the last pulse could be not a good chirp, so use the second from the back
    rx_error = np.array(data_ch0[0:N])
    return rx_error


def zca_whitening_matrix(X0):
    """
    potentially get rid of low effective sigma and compress the matrix: check p366 Gilbert Strang, "Linear Algebra"
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
    print(np.diag(cov_Y))  # Every time call this func will print. Should be all 1. It means basis are independent.

    #plt.matshow(abs(cov_Y))
    #plt.show()
    return Y


def tx_template(N, D, upsamp_rate):
    # 1. upsampling; 2. Shifting and high order; 3. downsampling; 4. ZCA
    '''
    j = 1j
    # fs = 28e6*upsamp_rate  # Sampling freq
    # N = upsamp_rate * N
    fs = 250e6  # Sampling freq
    # N = Norder_rep
    tc = N / fs  # T=N/fs#Chirp Duration
    t = np.linspace(0, tc, N)
    fc = 30e6
    bw = 20.0e6
    f0 = fc - bw / 2  # -10e6#40e6 # Start Freq
    f1 = fc + bw / 2  # 10e6#60e6# fs/2=1/2*N/T#End freq
    K = (f1 - f0) / tc  # chirp rate = BW/Druation
    phi_init = 0
    # win = np.blackman(N)
    # win=np.hamming(N)

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
    '''
    win = 1 #np.blackman(N)
    x_cx = coe_wavetable_4096.y_cx
    x_cx = np.multiply(x_cx, win)  # add window
    x_real = x_cx.real # only keep real part for the experiment.

    #y_calibration = readosc.readcsv(filename='data/output_cal_1.csv')
    #x_EQ = equalizer(x_real, y_calibration)  # Equalization filter

    # Add components group delay
    #s = touchstone('LBAND_FILTER_TOUCHSTONE_20MHZ_4096.S2P')
    #s21 = s[:, 1, 0]
    #x_cx_gd = fft.ifft(fft.fft(x_cx * s21))

    x_upsamp = upsampling(np.reshape(x_cx, (N, 1)), upsamp_rate)  # step 1: up-sampling
    x_upsamp = np.reshape(x_upsamp, N * upsamp_rate)

    x_cx_delay = 1j * np.ones([N * upsamp_rate, D]) #j * np.ones([N * upsamp_rate, D])
    # x_cx_order = j * np.ones([N * upsamp_rate])
    k0 = 0  # 180 # initial time delay for saving matrix space, take antenna cable into account
    k = 0  # delay tap
    order_idx = 1
    order = 1
    digital_filter_length = 300
    for i in range(D):
        x_cx_delay[:,i] = np.roll(x_upsamp, -0+100*i) # here  #x_cx_delay[:,i] = np.roll(x_cx, 20*i) # here

        #print (
        #    "digital_filter_length", digital_filter_length, "order_idx=", order_idx, "order=", order,

        #    'delay tap,k=', k)
        '''x_cx_order = np.power(abs(x_upsamp), order - 1) * x_upsamp
        x_cx_delay[:, i] = np.roll(x_cx_order, k + k0)
        if k == digital_filter_length - 1:
            print(k)
            k = 0
            order_idx += 1
            order = 2 * order_idx - 1
        else:
            k += 1
        '''
    x_cx_delay = downsampling(x_cx_delay, upsamp_rate)
    X = np.matrix(x_cx_delay)
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

    # Generate X for accelerating the gradient desce

    #M = int(N / 2)
    #X1 = X[int(N / 2) - M: int(N / 2) + M, :]
    for i in range(3):
        X1 = zca_whitening_matrix(X)
    # np.save('tx_template_order9_delay1_upsamp100_28MHz_x1', X1)

    return X1


def main(theta, N=4096, D=2, rx_error_sim=np.zeros([4096, 1]), itt=0):
    if itt == 0: # initial cancellation signal is set to zeros.
        UploadArb.UploadArb(np.zeros(N))
    start = timeit.default_timer()
    upsamp_rate = 1 #D #10#D
    # downsamp_rate = upsamp_rate

    X1 = tx_template(N, D, upsamp_rate)
    #X1 = coe_wavetable_4096.y_cx

    # Gradient Decentg
    readosc.readosc(itt,filename='output_1.csv') # take measurement on the Oscilloscope
    rx_error =readosc.readcsv(filename='output_1.csv')
    rx_error = np.expand_dims(rx_error, axis = 1)
    rx_error_cx = signal.hilbert(rx_error/(415e-3*2), axis=0)

    # rx_error = np.reshape(read_last_pulse("usrp_samples_loopback.dat", N), [N, 1])
    # rx_error_delay = np.roll(rx_error, -13)
    # rx_error = rx_error_delay
    # plt.plot(rx_error, '*-')
    # plt.plot(X[:,0], '*-')
    # plt.matshow(abs(np.cov(X, rowvar=False)))
    # plt.show()
    theta_cx_out, y_hat, cost_history = gd(theta, rx_error_cx, X1)
    #theta_cx_out, y_hat, cost_history = gd(theta_cx_in, rx_error_sim, X1)  # uncomment this line out when using simulated received signal

    #save_tx_canc('usrp_samples4096_chirp_28MHz_fixed_delayed.dat', N, y_hat)  # add FGPA tx delay inside this function
    ###########
    # Equalizing
    x_record = coe_wavetable_4096.y_cx.real
    y_record = readosc.readcsv(filename='data/x_canc_response.csv')
    y_hat_EQ = functions.equalizer(x_record, y_record, input=y_hat)  # step 2
    UploadArb.UploadArb(y_hat.real) # update the cancellation signal
    stop = timeit.default_timer()

    print('Time: ', stop - start)
    print('cost_history:', cost_history)
    print('theta = ',theta_cx_out)
    return theta_cx_out,cost_history#, y_hat, cost_history  # uncomment this line out when using simulated received signal


if __name__ == "__main__":  # Change the following code into the c++
    D = 20
    print('D = ', D)
    start = timeit.default_timer()
    mu, sigma = 0, 0.05
    np.random.seed(0)
    N = coe_wavetable_4096.N #4000  # This also limit the bandwidth. And this is determined by fpga LUT size.
    theta = (-1e-3 +1e-3 * 1j) * np.ones([D, 1])  # A small complex initial value. This should be a D * 1 column vector # +0.1j# *np.random.randn(1, 1) + 0.1j  # parameter to learn
    nitt = 100
    cost_history_all = np.zeros(nitt)
    for itt in range(nitt):
        print(itt)
        theta, cost_history = main(theta, N, D, itt =itt) # use this for real measurement
        cost_history_all[itt] = np.array(cost_history)[0][0]


    plt.plot(np.linspace(1,nitt, nitt), cost_history_all)
    plt.xlabel('Number of iteration')
    plt.ylabel('cost')
    plt.show()
    stop = timeit.default_timer()
    print('Time: ', stop - start)


