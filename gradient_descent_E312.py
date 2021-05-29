import numpy as np
import timeit
import os
import struct
from numpy import fft
from numpy.fft import fftshift
import skrf as rf
import time
import matplotlib.pyplot as plt
import readosc
import UploadArb
import coe_wavetable_4096 as coe
import setup
import functions
from scipy import signal
import dsp_filters_BPF


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
    return cost2


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


def gd(theta, rx_error_sim, X1):
    # gradient decent
    N = len(rx_error_sim)
    M = int(len(X1) / 2)  # N/2
    eta = setup.eta  # learning rate
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

    y_hat = np.dot((X1), theta)  # y_hat = prediction for cancellation signal
    cost_history = 10 * np.log10(abs(cal_cost(X1, rx_error_sim)))
    return theta, y_hat, cost_history


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


def shift_tap(k):
    shift = setup.delay_0 + setup.delay_step * k
    #if 50 < shift < 1000: #0.6 meter/tap
    if 0 < shift < 1000:  # 0.6 meter/tap
        shift1 = shift + 1000
    else:
        shift1 = shift
    return shift1


def X_matrix(x, K, Q, upsamp_rate, debug=False):
    # This function make the template matrix
    # K is the total delay or the filter total length
    # Q is number of different high order terms. e.g. The highest odd order is 2Q-1.
    #X = 1j * np.ones([N * upsamp_rate, D])
    X = np.array([])
    x_sub0 = 1j * np.ones([N * upsamp_rate, K+1]) # initial value
    x_sub1 = 1j * np.ones([N * upsamp_rate, K+1]) # initial value
    for q in range (1, Q+1):
        order = 2 * q -1
        #if q == 1: # If the order is 2*q -1 = 1, just make the matrix X with all delays.
        if q == 1:  # If the order is 2*q -1 = 1, just make the matrix X with all delays.
            for k in range (0, K+1):
                shift = shift_tap(k)  # setup.delay_0 + setup.delay_step * k #shift_tap(k) # range selection
                x_delay = np.roll(x, shift)
                x_sub0[:, k] = np.power(abs(x_delay.real), order - 1) * x_delay.real
                if debug: print("digital_filter_length", K, "q(order_idx)=", q, "order=", order, 'delay tap,k=', k)
            X = x_sub0
        if q > 1:  # If there are more orders, generate delays with higher order. This for loop creats a sub-matrix
            # for a order = 2q-1 with all delays
            for k in range (0, K+1):
                shift = setup.delay_0 + setup.delay_step * k  #setup.delay_0 + setup.delay_step * k # no range selection #shift_tap(k)
                x_delay = np.roll(x, shift)
                x_sub1[:, k] = np.power(abs(x_delay.real), order - 1) * x_delay.real
                if debug: print("digital_filter_length", K, "q(order_idx)=", q, "order=", order, 'delay tap,k=', k)
            X = np.concatenate((X, x_sub1), axis = 1) # If there are more orders, stack them as submatrix with order = 1.
    return np.matrix(X)



def tx_template(N, K, Q, D, upsamp_rate):
    # 1. upsampling; 2. Shifting and high order; 3. downsampling; 4. ZCA
    win = 1 #np.blackman(N)
    x_cx = coe.y_cx
    x_cx = np.multiply(x_cx, win)  # add window
    #x_cx = np.load(setup.file_tx) # using T.L loopback as template basis
    x_upsamp = upsampling(np.reshape(x_cx, (N, 1)), upsamp_rate)  # step 1: up-sampling
    x_upsamp = np.reshape(x_upsamp, N * upsamp_rate)
    # parameters
    X = X_matrix(x_upsamp, K=K, Q=Q, upsamp_rate=upsamp_rate)

    X = downsampling(X, upsamp_rate)

    for i in range(1):
        X1 = zca_whitening_matrix(X)
    X1 = np.matrix(signal.hilbert(X1.real, axis = 0))
    return X1


def main(theta, N=4096, K=0, Q=1, D=1, rx_error_sim=np.zeros([4096, 1]), itt=0, simulation_flag=False, EQ_flag=False):
    start = timeit.default_timer()
    upsamp_rate = setup.upsamp_rate #D
    X1 = tx_template(N, K, Q, D, upsamp_rate)
    #################################
    # Step 1: Get the residual signal
    #################################
    if not simulation_flag:
        if itt == 0:  # initial cancellation signal is set to zeros.
            UploadArb.UploadArb(np.zeros(N))
        readosc.readosc(itt,filename='output_1.csv') # take measurement on the Oscilloscope
        rx_error = 100*np.reshape(readosc.readcsv(filename='output_1.csv'), [N,1])
    else:
        rx_error = rx_error_sim.real
    rx_error_cascade = np.vstack((rx_error,rx_error))
    rx_error_cx = signal.hilbert(rx_error_cascade, axis=0)
    rx_error_cx = rx_error_cx[-1-N+1:]  # Take the second part of the Hilbert transform due to the first several points are bad
    rx_error_cx = functions.dcblocker(rx_error_cx)
    #rx_error_cx = rx_error
    #########################
    # Step 2: Gradient Decent
    #########################
    theta_cx_out, y_hat, cost_history = gd(theta, rx_error_cx, X1)

    ####################
    # Step 3: Equalizing
    ####################
    if not simulation_flag:
        if EQ_flag:
            x_record = coe.y_cx.real  # preset signal
            y_record = setup.y_EQ  # Pre-recorded signal
            y_hat_EQ = functions.equalizer(x_record, y_record, input=y_hat)  # step 2
            UploadArb.UploadArb(-y_hat_EQ.real) # update the cancellation signal
        else:
            UploadArb.UploadArb(np.roll(np.array(y_hat.real),0))
            
    ############
    # debug info
    ############
    stop = timeit.default_timer()
    print('Time: ', stop - start)
    print('cost_history:', cost_history)
    print('theta = ',theta_cx_out)
    return theta_cx_out, y_hat, cost_history  # uncomment this line out when using simulated received signal


if __name__ == "__main__":  # Change the following code into the c++
    N = setup.N  # 4000  # This also limit the bandwidth. And this is determined by fpga LUT size.
    K = setup.K  # Max delay, length of the filter
    Q = setup.Q  # max order = 2*Q-1
    D = (K+1) * Q  # Plus one due to there is original non-delayed basis.
    simulation_flag = setup.simulation_flag
    EQ_flag = setup.EQ_flag
    print('D = ', D)
    y_hat = np.zeros([N, 1])
    if simulation_flag:
        # y is the simulated received signal
        y = np.reshape(setup.y_sim, [N,1]) #np.reshape(coe.y_cx, [N, 1]) # initial received signal for simulation.
        y_cx = signal.hilbert(y, axis=0)
        y = functions.dcblocker(y_cx) # uncomment this when using chirp rather than low freq sine wave

        #y0 = functions.dcblocker(y_cx) # uncomment this when using chirp rather than low freq sine wave
        #y = np.squeeze(y0)
        #y = dsp_filters_BPF.run(y)
        #y = y.reshape([N, 1])
    t = coe.t*1e6
    start = timeit.default_timer()
    theta = (-1e-3 + 1e-3 * 1j) * np.ones([D, 1])  # A small complex initial value. This should be a D * 1 column vector # +0.1j# *np.random.randn(1, 1) + 0.1j  # parameter to learn


    nitt = setup.nitt
    cost_history_all = np.zeros(nitt)
    for itt in range(nitt):
        print(itt)
        if simulation_flag:
            rx_error_sim = y + y_hat
            #rx_error_sim = np.squeeze(rx_error_sim)
            #rx_error_sim = dsp_filters_BPF.run(rx_error_sim)  # uncomment this line out when using simulated received signal
            #rx_error_sim = np.reshape(rx_error_sim, [N,1])
            theta, y_hat, cost_history = main(theta=theta, N=N, K=K, Q=Q,  D=D, rx_error_sim=rx_error_sim, itt=itt, simulation_flag=True)  # uncomment this line out when using simulated received signal
        else:
            theta, y_hat, cost_history = main(theta=theta, N=N, K=K, Q=Q, D=D, itt =itt, simulation_flag=False, EQ_flag=EQ_flag) # use this for measurement
        cost_history_all[itt] = cost_history#[0]

    #rx_error_sim = np.array(rx_error_sim)
    #np.save('rx_error_sim', rx_error_sim)
    plt.plot(np.linspace(1,nitt, nitt), cost_history_all, '.-')
    plt.xlabel('Number of iteration')
    plt.ylabel('cost [dB]')

    if simulation_flag:
        plt.figure(2)
        plt.plot(t, y, t, -y_hat)
        #plt.plot(t, rx_error_sim)
        plt.xlabel('Time [$\mu s$]')
        plt.ylabel('Signals')
        #plt.legend(['y', '$\hat{y}$', '$y-\hat{y}$'])
        plt.legend(['y', '$\hat{y}$'])
        plt.figure(3)
        plt.plot(t, rx_error_sim)
        plt.xlabel('Time [$\mu s$]')
        plt.ylabel('Residual')
        plt.figure(4)
        functions.plot_freq_db(coe.freq/1e6, y.real, color='b')
        #plt.figure()
        functions.plot_freq_db(coe.freq/1e6, rx_error_sim.real, color='k')
        plt.xlabel('Frequency [MHz]')
        plt.ylabel('Amplitude [dB]')
        plt.title('Residual Frequency Response')
        #plt.ylim(-10, 120)

        # Pulse compression after cancellation
        win = np.blackman(N)
        tx = coe.y_cx.real # np.load(file=setup.file_tx)
        PC1 = functions.PulseCompr(rx=np.squeeze(np.array(rx_error_sim)), tx=signal.hilbert(tx, axis= 0), win=win)

        # Pulse compression before cancellation
        PC2 = functions.PulseCompr(rx=np.squeeze(np.array(y)), tx=signal.hilbert(tx, axis= 0), win=win)
        plt.figure(5)
        plt.plot(fftshift(coe.distance), fftshift(PC1), 'k*-', fftshift(coe.distance), fftshift(PC2), 'b')
        plt.xlim((-500, 500))
        #plt.ylim((-30, 150))
        plt.title('Pulse Compression')
        plt.xlabel('Distance in meter')
        plt.ylabel('Power in dB')
        plt.legend(['After Leakage Cancellation', 'Before Leakage Cancellation'])
        plt.grid()
    plt.show()
    stop = timeit.default_timer()
    print('Time: ', stop - start)


