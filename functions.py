import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fftshift
import setup


def dcblocker(x):
    X = np.fft.fft(x, axis = 0)
    #X[-1] = 0
    X[0:20] = 0
    X[-1 - 20:] = 0

    #X[0:100] = 0
    #X[-1 - 100:] = 0
    y = np.fft.ifft(X,  axis = 0)
    return y


def upsampling(x, upsamp_rate):
    # Actually no need. Just use higher fs to generate better template digitally is good enough.
    # This is just a one-dimensional interpolation.
    # https://dsp.stackexchange.com/questions/14919/upsample-data-using-ffts-how-is-this-exactly-done
    # FFT upsampling method
    N = x.shape[0]
    D = 1 #x.shape[1]
    # To frequency domain
    X = np.fft.fft(x, axis=0)
    # Add taps in the middle
    A1 = X[0:int(N / 2)]
    A2 = np.zeros([(upsamp_rate - 1) * N])
    A3 = X[int(N / 2):N]
    XX = np.concatenate((A1, A2, A3))
    # To time domain
    xx = upsamp_rate * np.fft.ifft(XX, axis=0)
    # plt.plot(np.linspace(0,1, xx.shape[0]), xx)
    # plt.plot(np.linspace(0,1, xx.shape[0]), xx ,'ko')
    # plt.plot(np.linspace(0,1, N), x, 'y.')
    # plt.show()
    x_upsamp =xx# np.reshape(xx, (N * upsamp_rate, D))  # change back to 1-D
    return x_upsamp


def downsampling(x, downsamp_rate):
    #N = x.shape[0]
    #D = x.shape[1]
    x_downsamp = x[::downsamp_rate]
    # plt.plot(np.linspace(0,1, x_downsamp.shape[0]), x_downsamp)
    # plt.plot(np.linspace(0,1, x_downsamp.shape[0]), x_downsamp ,'ko')
    # plt.plot(np.linspace(0,1, N), x, 'y.')
    # plt.show()
    return x_downsamp


def equalizer(x, y, input, scale=500):
    #####################
    # Equalization filter:
    # Equalizers are used to render the frequency response and flat it from end-to-end
    # x is the ideal signal. y is the received signal distorted by the channel, y is the response of x.
    # input is the signal wanted to be sent out. The input will be equalized by the equalizing filter.
    # Usage: 1. Send a signal x and record its response y. 2. Calculate the equalizing filter. 3. Apply the equalizing filter to a new signal.


    #####################
    #X = np.fft.fft(x)
    #Y = np.fft.fft(y)
    input = np.squeeze(np.array(input))
    y = y /max(abs(y))
    X = np.fft.fft(x) # https://stackoverflow.com/questions/52387673/what-is-the-difference-between np-fft-fft-and np-fft-rfft
    Y = np.fft.fft(y)
    # Set the signal out of the band to a fixed number to avoid too small value. The small value will give spike if it is inversed.
    Y[20*np.log10(abs(Y))<30] = 1000 # tune the two params: abs(Y)< a and = b for smooth EQ output in freq domain. Should be no discontinuity in freq.
    INPUT = np.fft.fft(input)
    H = np.divide(Y, X)
    H_inv = 1/H
    #H_inv[0:400]=1e-10*np.ones([400])
    #H_inv[800:2501] = 1e-10*np.ones([1701])
    #plt.close()
    #plt.plot(20 * np.log10(X))
    #plt.plot(20 * np.log10(Y))
    #plt.show()
    #print(np.fft.ifft(H_inv).real.mean())
    #print(np.fft.ifft(H_inv).imag.mean())
    X_EQ = np.multiply(H_inv, X)
    X_EQ[0] = 0 # setting the bin zero(DC component to zero) to get rid of DC offset. It is not zero due to noise and randomness.
    x_EQ = np.fft.ifft(X_EQ)
    x_EQ = x_EQ/max(abs(x_EQ))
    #
    INPUT_EQ = np.multiply(H_inv, INPUT)
    INPUT_EQ[0] = 0 # setting the bin zero(DC component to zero) to get rid of DC offset. It is not zero due to noise and randomness.
    input_EQ = np.fft.ifft(INPUT_EQ)
    if scale == 0:
        input_EQ = input_EQ / max(abs(input_EQ))
    else:
        input_EQ = input_EQ / scale  # Scaled by 500 times by default for tune the magnitude for gradient decent

    return input_EQ


def PulseCompr(rx,tx,win, unit = 'log'):
    # Mixer method pulse compression; Return a log scale beat frequency signal.
    a = np.multiply(rx,win)  #np.power(win, 10)#np.multiply(win,win) # Add window here
    b = np.multiply(tx,np.power(win, 0))  #np.power(win, 10)#np.multiply(win,win)#tx
    mix = b * np.conj(a)  # 1. time domain element wise multiplication.
    pc = np.fft.fft(mix)  # 2. Fourier transform.
    # Add LPF
    #pc_timedomain = np.fft.ifft(pc)
    #pc_LPF = dsp_filters.main(signal=pc_timedomain, order=6, fs=250e6, cutoff=6e6, duration=1.6e-6)
    #pc_LPF_freqdomain = np.fft.fft(pc_LPF)
    # match filter method
    #A = np.fft.fft(a)
    #B = np.fft.fft(b)
    #pc_mf = np.fft.ifft(np.multiply(B, np.conj(A)))
    if unit == 'log':
        pc = 20 * np.log10(abs(pc))
    if unit =='linear':
        pc = pc
    return pc


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
        S), Mx  # add a np.conj() because when I compare to the np.cov() the result only be the same when adding the conjucate... strange.


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
    #print(np.diag(cov_Y))  # Every time call this func will print. Should be all 1. It means basis are independent.

    # plt.matshow(abs(cov_Y))
    # plt.show()
    return Y


def channel_est(psi_orth, y_cx_received):
    # Calculate weights
    # 1. Make H matrix
    N= len(y_cx_received)
    '''
    H = psi_orth
    D = 100
    for idx in range(-D, D):
        psi_orth_delay_roll = np.roll(psi_orth, idx*4) #*4
        H = np.column_stack((H, psi_orth_delay_roll))
        #print(H)
    '''
    '''
    psi_orth_delay_0 = np.concatenate((np.zeros(0), psi_orth[:N-0]), axis=0)
    psi_orth_delay_1 = np.concatenate((np.zeros(1), psi_orth[:N-1]), axis=0)
    psi_orth_delay_2 = np.concatenate((np.zeros(2), psi_orth[:N-2]), axis=0)
    psi_orth_delay_3 = np.concatenate((np.zeros(3), psi_orth[:N-3]), axis=0)
    psi_orth_delay_4 = np.concatenate((np.zeros(4), psi_orth[:N-4]), axis=0)
    psi_orth_delay_5 = np.concatenate((np.zeros(5), psi_orth[:N-5]), axis=0)
    psi_orth_delay_6 = np.concatenate((np.zeros(6), psi_orth[:N-6]), axis=0)
    psi_orth_delay_7 = np.concatenate((np.zeros(7), psi_orth[:N-7]), axis=0)
    psi_orth_delay_8 = np.concatenate((np.zeros(8), psi_orth[:N-8]), axis=0)
    psi_orth_delay_9 = np.concatenate((np.zeros(9), psi_orth[:N-9]), axis=0)

    H= np.transpose([psi_orth_delay_0,psi_orth_delay_1,psi_orth_delay_2,psi_orth_delay_3,psi_orth_delay_4,psi_orth_delay_5,
        psi_orth_delay_6,psi_orth_delay_7,psi_orth_delay_8,psi_orth_delay_9])
    '''
    #print('Shape of H is',H.shape)

    H = X_matrix(psi_orth, K = setup.K, Q= setup.Q, upsamp_rate= setup.upsamp_rate, debug=False)
    for i in range(3):
        H = zca_whitening_matrix(H)
    # 2. Make y vector
    y = np.transpose(y_cx_received[np.newaxis])
    #print('Shape of y is', y.shape)

    # 3. Calculate channel
    R_H = np.dot(np.matrix.getH(H),H) # Covariance Matrix of H
    #print('Shape of R_H is', R_H.shape)
    R_yH = np.dot(np.matrix.getH(H),y)# Cross-covariance Matrix of {y,H}
    #print('Shape of R_yH is', R_yH.shape)
    c = np.dot(np.linalg.inv(R_H),R_yH)
    #print('Shape of c is', c.shape)
    return c, H


def plot_freq_db(freq, x, color='b', normalize=False):
    # This function plots the input in frequency domain in dB
    X = np.fft.fft(x,axis = 0)
    #freq = np.fft.fftfreq(len(x))
    X_log = 20*np.log10(np.abs(X))
    X_log_normalize = X_log - max(X_log)
    if normalize:
        plt.plot(fftshift(freq), fftshift(X_log_normalize), color=color)
    else:
        plt.plot(fftshift(freq), fftshift(X_log), color = color)
    plt.grid(b= True)
    return X


def plot_freq_distance(distance, pc):
    fig, ax = plt.subplots()
    ax.plot(np.fft.fftshift(distance), np.fft.fftshift(pc), '*-')
    plt.xlabel('Distance [m]')
    secax = ax.secondary_xaxis('top', functions=(distance2freq, freq2distance))
    secax.set_xlabel('Frequency [MHz]')
    plt.grid()
    plt.ylabel('Amplitude [dB]')
    plt.title('Pulse Compression')


def distance2freq(distance):
    # https://matplotlib.org/stable/gallery/subplots_axes_and_figures/secondary_axis.html
    coe = Coe()
    k = coe.k # Pick your k value! This k cannot change by different setup. It is fixed by default
    c = 3e8
    freq = distance / c * k * 2.0
    return freq / 1e6  # MHz


def freq2distance(freq):
    coe = Coe()
    k = coe.k # Pick your k value! This k cannot change by different setup. It is fixed by default
    c = 3e8
    distance = c * freq * 1e6 / k / 2.0
    return distance


def normalize(x):
    #  Normalize the signal in dB.
    #x_abs = abs(x)
    y = x - max(x)
    return y


def shift_tap(k):
    shift = setup.delay_0 + setup.delay_step * k
    if 100 < shift < 400:
        shift1 = shift + 400
    else:
        shift1 = shift
    return shift1


def X_matrix(x, K, Q, upsamp_rate, debug=False):
    # This function make the template matrix
    # K is the total delay or the filter total length
    # Q is number of different high order terms. e.g. The highest odd order is 2Q-1.
    #X = 1j * np.ones([N * upsamp_rate, D])
    X = np.array([])
    N = len(x)
    x_sub0 = 1j * np.ones([N * upsamp_rate, K+1]) # initial value
    x_sub1 = x_sub0 # initial value
    for q in range (1, Q+1):
        order = 2 * q -1
        #if q == 1: # If the order is 2*q -1 = 1, just make the matrix X with all delays.
        if q == 1:  # If the order is 2*q -1 = 1, just make the matrix X with all delays.
            for k in range (0, K+1):
                shift = shift_tap(k) # range selection
                x_delay = np.roll(x, shift)
                x_sub0[:, k] = np.power(abs(x_delay), order - 1) * x_delay
                if debug: print("digital_filter_length", K, "q(order_idx)=", q, "order=", order, 'delay tap,k=', k)
            X = x_sub0
        if q > 1:  # If there are more orders, generate delays with higher order. This for loop creats a sub-matrix
            # for a order = 2q-1 with all delays
            for k in range (0, K+1):
                shift = setup.delay_0 + setup.delay_step * k # no range selection #shift_tap(k)
                x_delay = np.roll(x, shift)
                x_sub1[:, k] = np.power(abs(x_delay), order - 1) * x_delay
                if debug: print("digital_filter_length", K, "q(order_idx)=", q, "order=", order, 'delay tap,k=', k)
            X = np.concatenate((X, x_sub1), axis = 1) # If there are more orders, stack them as submatrix with order = 1.
    return np.matrix(X)


def cal_model_parameter(H, y):
    # From digital_predistortion_v2_analitical.py
    # Calculate the parameter analytically
    y = np.transpose(y[np.newaxis])
    print('Shape of y is', y.shape)
    # Calculate model parameter analytically.
    R_H = np.dot(np.matrix.getH(H), H)  # Covariance Matrix of H
    print('Shape of R_H is', R_H.shape)
    R_yH = np.dot(np.matrix.getH(H), y)  # Cross-covariance Matrix of {y,H}
    print('Shape of R_yH is', R_yH.shape)
    w = np.dot(np.linalg.inv(R_H), R_yH)
    return w


def phi_gen(x, order, delay):
    # From digital_predistortion_v2_analitical.py
    # calculate basis vector matrix
    D = order * delay
    phi = 1j * np.ones([len(x), D])

    # x_cx_order = j * np.ones([N * upsamp_rate])
    # q0 = 0  # 180 # initial time delay for saving matrix space, take antenna cable into account
    # q = 0  # delay tap
    digital_filter_length = delay

    for q in range(delay):
        for k in range(order):
            idx = q * order + k
            power_order = (2 * k+1)
            x_cx_order = np.power(abs(x), power_order)* x
            # x_cx_order = np.power(x, power_order) * x
            x_cx_delay = np.roll(x_cx_order, q)
            phi[:, idx] = x_cx_delay
            print("digital_filter_length", digital_filter_length, "idx=", idx, "k=", k, "power order =", power_order,
                  ', delay tap=', q)

    # for i in range(3):
    #    phi = zca_whitening_matrix(phi)
    return phi


class Coe:
    def __init__(self, fc=50e6, bw=20e6, fs=250e6, N=1024):
        # initialize the object’s state and assign values to the data members
        self.c = 3e8
        self.fs = fs  # 250e6 #56e6 #1000e6 #250e6  # Sampling freq
        self.bw = bw
        self.N = N
        self.T = self.N / self.fs
        self.k = self.bw / self.T
        self.fc = fc
        ####################################
        c = self.c
        N = self.N
        fs = self.fs
        T = self.T  # T=N/fs#Chirp Duration
        t = np.linspace(0, T, N)
        fc = self.fc
        bw = self.bw  # 20e6#45.0e5
        win = 1
        j = 1j
        f0 = fc - bw / 2  # -10e6#40e6 # Start Freq
        f1 = fc + bw / 2  # 10e6#60e6# fs/2=1/2*N/T#End freq
        #print('f0 = ', f0 / 1e6, 'MHz;', 'f1=', f1 / 1e6, 'MHz')
        k = self.k  # (f1 - f0) / T
        phi0 = -np.pi / 2  # Phase
        self.freq = np.fft.fftfreq(N, d=1. / fs)
        self.distance = c * self.freq / k / 2.0  # = c/(2BW), because need an array of distance, so use freq to represent distance.
        ##################
        # Create the chirp
        ##################
        y = np.sin(2 * np.pi * (f0 * t + k / 2 * np.power(t, 2)))  # use this for chirp generation
        yq = np.sin(phi0 + 2 * np.pi * (f0 * t + k / 2 * np.power(t, 2)))  # use this for chirp generation
        y_cx_0 = y + j * yq
        ##################
        # Create the sine
        ##################
        y_s = np.sin(1 * 2 * np.pi * fs / N * t)  # + np.sin(4 np.pi*fs/N*t)# just use LO to generate a LO. The
        yq_s = np.sin(1 * 2 * np.pi * fs / N * t - np.pi / 2)  # + np.sin(4 np.pi*fs/N*t np.pi/2)
        self.y_cx_sine = y_s + j * yq_s
        fo = 50e6
        y_s2 = np.sin(1 * 2 * np.pi * fo * t)  # + np.sin(4 np.pi*fs/N*t)# just use LO to generate a LO. The
        yq_s2 = np.sin(1 * 2 * np.pi * fo * t - np.pi / 2)  # + np.sin(4 np.pi*fs/N*t np.pi/2)
        self.y_cx_sine2 = y_s2 + j * yq_s2

        self.y_cx = y_cx_0  # y_cx_0 #y_cx_sine #y_cx_0 #y_cx_0 #y_cx_sine2
        # plt.plot(freq/1e6, 20 np.log10(abs np.fft.fft(y_cx.real))))
        # plt.grid()
        # plt.xlabel('Frequency [MHz]')

        delay = 300  # 30*7.5 = 225 meter
        # y_cx_0_delay = np.concatenate( np.zeros(100), y_cx_0[:N-100]), axis=0)
        y_cx_0_delay = np.roll(np.multiply(y_cx_0, win),
                                  delay)  # np.concatenate((y_cx_0[N-delay-1:-1], y_cx_0[:N-delay]), axis=0)
        y_cx_combine = 0.5 * y_cx_0 + 0.05 * y_cx_0_delay

        # A noise signal
        mu, sigma = 0, 0.000000005 # 0.5 #0.000000005
        np.random.seed(0)
        self.y_noise = np.random.normal(mu, sigma, N)
