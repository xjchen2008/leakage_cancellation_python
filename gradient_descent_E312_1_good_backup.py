import numpy as np
import timeit
import os
import struct
from numpy import fft
import time
import matplotlib.pyplot as plt


def sample_cov(X):
    # Check the math here https://www.itl.nist.gov/div898/handbook/pmc/section5/pmc541.htm
    # Estimation of covariance matrices: https://en.wikipedia.org/wiki/Covariance_matrix
    X = np.matrix(X)
    N = X.shape[0] # rows of X: number of observations
    D = X.shape[1] # columns of X: number of variables
    mean_col = 1j*np.ones(D) # has to define a complex number for keeping the imaginary part
    for col_indx in range(D):
        mean_col[col_indx] = np.mean(X[:,col_indx])
    Mx = X - mean_col # Zero mean matrix of X
    S = np.dot(Mx.H,Mx) / (N-1) # sample covariance matrix
    return np.conj(S), Mx # add a np.conj() because when I compare to the numpy.cov() the result only be the same when adding the conjucate... strange.


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
    sigma0 = np.cov(X0, rowvar=False) # [D x D]
    #print(sigma0)
    sigma, Mx = sample_cov(X0)
    #print(sigma)

    XhX = np.dot(Mx.H, Mx) # (N-1)*sigma should be the same but there is a conjugate difference. Don't know why...
    # Singular Value Decomposition. X = U * np.diag(S) * V
    U,S,Vh = np.linalg.svd(XhX)
        # U: [D x D] eigenvectors of sigma.
        # S: [D x 1] eigenvalues of sigma.
        # V: [D x D] transpose of U
    # Whitening constant: prevents division by zero
    epsilon = 1e-1000
    ZCAMatrix = np.sqrt(N-1) * np.dot(Vh.H, np.dot(np.diag(1.0 / np.sqrt(S + epsilon)), Vh))  # [M x M]

    Y = np.dot(Mx, ZCAMatrix)
    cov_Y = np.cov(Y, rowvar=False)
    print(np.diag(cov_Y))

    #plt.matshow(abs(cov_Y))
    #plt.show()
    return Y


def array2tuple(array, D):
    theta_tuple = ()
    for i in range(D):
        arr = [(np.real(array[i]), np.imag(array[i]))]
        theta_tuple = theta_tuple + tuple(map(tuple,arr))
    return theta_tuple


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
    rx_sig = -x_sig[0::2] + 1j*x_sig[1::2] # Important! Here I added a negtive sign here for calibrate the tx rx chain. There is a flip of sign somewhere but I cannot find.
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
    eta = 0.5# 1  # learning rate
    m = len(rx_error_sim)
    c2 =1#0.45 # this is a amplitude calibration coefficient = Rx/Tx, this will also make the convergence faster.
    theta = theta -(1.0 / m) * eta * np.dot(np.conj(X.T), rx_error_sim / c2)
    #theta = theta - (1.0 / m) * eta * np.dot(np.real(X.T), np.real(rx_error_sim)/ c2)
    y_hat = np.dot((X), theta) # y_hat = prediction for cancellation signal
    cost_history = 10*np.log10(cal_cost(rx_error_sim))
    return theta, y_hat, cost_history


def upsampling(x, upsamp_rate):
    # Actually no need. Just use higher fs to generate better template digitally is good enough.
    # This is just a one-dimensional interpolation.
    # https://dsp.stackexchange.com/questions/14919/upsample-data-using-ffts-how-is-this-exactly-done
    # FFT upsampling method

    N = x.shape[0]
    D = x.shape[1]
    # To frequency domain
    X = fft.fft(x,axis = 0)
    # Add taps in the middle
    A1= X[0:N/2,:]
    A2= np.zeros([(upsamp_rate-1)*N,D])
    A3= X[N/2:N,:]
    XX = np.concatenate((A1,A2,A3))
    # To time domain
    xx = upsamp_rate * fft.ifft(XX,axis = 0)
    #plt.plot(np.linspace(0,1, xx.shape[0]), xx)
    #plt.plot(np.linspace(0,1, xx.shape[0]), xx ,'ko')
    #plt.plot(np.linspace(0,1, N), x, 'y.')
    #plt.show()
    x_upsamp = np.reshape(xx, (N*upsamp_rate,D)) # change back to 1-D
    return x_upsamp


def downsampling(x, downsamp_rate):
    N = x.shape[0]
    D = x.shape[1]
    x_downsamp = x[::downsamp_rate]
    #plt.plot(np.linspace(0,1, x_downsamp.shape[0]), x_downsamp)
    #plt.plot(np.linspace(0,1, x_downsamp.shape[0]), x_downsamp ,'ko')
    #plt.plot(np.linspace(0,1, N), x, 'y.')
    #plt.show()
    return x_downsamp


def tx_template(N, D, upsamp_rate):
    j = 1j
    #fs = 28e6*upsamp_rate  # Sampling freq
    #N = upsamp_rate * N
    fs = 28e6  # Sampling freq
    N = N
    tc =  N / fs  # T=N/fs#Chirp Duration
    t = np.linspace(0, tc, N)
    f0 = -14e6  # Start Freq
    f1 = 14e6  # fs/2=1/2*N/T#End freq
    K = (f1 - f0) / tc  # chirp rate = BW/Druation
    # win = np.blackman(N)
    # win=np.hamming(N)
    win=1
    # Sine wave
    #x0 = 1.0 * np.sin(2 * np.pi * fs / N * t)
    #xq0 = 1.0 * np.sin(2 * np.pi * fs / N * t - np.pi / 2)

    # Chirp
    x0 = 1 * np.sin(2 * np.pi * (f0 * t + K / 2 * np.power(t, 2)))  # use this for chirp generation
    xq0 = 1 * np.sin(2 * np.pi * (f0 * t + K / 2 * np.power(t, 2)) - np.pi / 2)  # use this for chirp generation
    # Square Wave
    #x0 = np.concatenate((np.zeros(2048),np.ones(2048)))
    #xq0 = np.concatenate((np.zeros(2048),np.ones(2048)))

    x_cx = x0 + j * xq0
    x_cx = np.multiply(x_cx, win) # add window
    x_upsamp = upsampling(np.reshape(x_cx, (N,1)), upsamp_rate)  # step 1: up-sampling
    x_upsamp = np.reshape(x_upsamp,N*upsamp_rate)

    x_cx_delay = j*np.ones([N*upsamp_rate,D])
    for i in range (D):
        #phi_init = 2 * np.pi / N / (D-1) * i * 100
        #x0 = 1 * np.sin(2 * np.pi * (f0 * t + K / 2 * np.power(t, 2)) + phi_init)  # use this for chirp generation
        #xq0 = 1 * np.sin(2 * np.pi * (f0 * t + K / 2 * np.power(t, 2)) - np.pi / 2 + phi_init)  # use this for chirp generation
        #x_cx_delay[:,i] = x0 + j * xq0
        x_cx_delay[:,i] = np.roll(x_upsamp, 0+1*i) # here  #x_cx_delay[:,i] = np.roll(x_cx, 20*i) # here


    #x_cx_delay[:, 0] = np.roll(x_cx, 0)#np.zeros(N)
    #x_cx_delay[:, 1] = np.roll(x_cx, 100)  # x_cx

    x_cx_delay = downsampling(x_cx_delay, upsamp_rate)
    X = np.matrix(x_cx_delay)
    return X


def save_tx_canc(filename, N,y_hat):
    # save y_hat
    #y_hat = np.concatenate((y_hat[N - 2028+2048:], y_hat[:N - 2028+2048]), axis=0) # fixed delay for tx chain from FPGA to RF out which need to measured for cancellation path
    y_hat = np.concatenate((y_hat[N:], y_hat[:N]),axis=0)  # fixed delay for tx chain from FPGA to RF out which need to measured for cancellation path
    y_cxnew = np.around(32767 * y_hat )  # numpy.multiply(y_cx,win) 6.9

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
    Amp = 1
    phi_init1 = 0/180#-105.0 / 180.0 * np.pi
    phi_init2 = 0/180#-105.0 / 180.0 * np.pi
    dc_offset = 0

    # Sine wave
    #y0 = Amp * np.sin(2 * np.pi * fs / N * t)
    #yq0 = Amp * np.sin(2 * np.pi * fs / N * t - np.pi / 2)

    #y0 = Amp * np.sin(2 * np.pi * fs / N * t + phi_init1) + dc_offset # + numpy.sin(4*numpy.pi*fs/N*t)# just use LO to generate a LO. The
    #yq0 = Amp * np.sin(2 * np.pi * fs / N * t - np.pi / 2 + phi_init2) + dc_offset  # + numpy.sin(4*numpy.pi*fs/N*t-numpy.pi/2)

    # Chirp
    y0 = Amp * np.sin(phi_init1 + 2 * np.pi * (f0 * t + K / 2 * np.power(t, 2))) + dc_offset  # use this for chirp generation
    yq0 = Amp * np.sin(phi_init2 + 2 * np.pi * (f0 * t + K / 2 * np.power(t, 2)) - np.pi / 2) + dc_offset  # use this for chirp generation

    # Square
    #y0 = np.concatenate((np.zeros(2048),np.ones(2048)))
    #yq0 = np.concatenate((np.zeros(2048),np.ones(2048)))
    y = y0 + j * yq0
    y = np.multiply(y, win)
    #y_cx_delay = np.concatenate((y[N-0:], y[:N-0]), axis=0)

    y = np.reshape(y, [N, 1])
    return y


def read_last_pulse(filename, N):
    nsamples = N
    channels = 2
    rubish = 0
    # Read latest pulse
    data_ch0, data_ch1 = last_pulse(filename, nsamples * channels, rubish)
    rx_error = np.array(data_ch0[0:N])
    return rx_error


def main(theta, N=4096, D=2, rx_error_sim = np.zeros([4096, 1])):
    print('D = ', D)
    start = timeit.default_timer()

    upsamp_rate = D
    downsamp_rate = upsamp_rate

    X0 = tx_template(N,D, upsamp_rate) # Step 1: up-sampling and time delay
    X1 = zca_whitening_matrix(X0) # step 2: whitening
    X = zca_whitening_matrix(X1)


    # Gradient Decentg
    rx_error = np.reshape(read_last_pulse("usrp_samples_loopback.dat",N), [N,1])
    #rx_error_delay = np.roll(rx_error, -13)
    #rx_error = rx_error_delay
    #plt.plot(rx_error, '*-')
    #plt.plot(X[:,0], '*-')
    #plt.matshow(abs(np.cov(X, rowvar=False)))
    #plt.show()
    theta_cx_in = np.zeros([D,1])
    for i in range(D):
        theta_cx_in[i] = np.array(theta)[i][0] + 1j*np.array(theta)[i][1]

    #theta_cx_out, y_hat, cost_history = gd(theta_cx_in, rx_error, X)
    theta_cx_out, y_hat, cost_history = gd(theta_cx_in, rx_error_sim, X) #uncomment this line out when using simulated received signal

    save_tx_canc('usrp_samples4096_chirp_28MHz_fixed_delayed.dat', N, y_hat) # add FGPA tx delay inside this function
    stop = timeit.default_timer()

    print('Time: ', stop - start)
    print('cost_history:', cost_history)
    print('Theta_cx_out:', theta_cx_out)
    theta_real_out = []
    theta_imag_out = []
    theta_tuple_out = ()
    for i in range(D):
        theta_real_out.append(float(np.matrix.tolist(np.real(theta_cx_out.T))[0][i]))
        theta_imag_out.append(float(np.matrix.tolist(np.imag(theta_cx_out.T))[0][i]))
        arr = [(theta_real_out[i],theta_imag_out[i] )]
        theta_tuple_out = theta_tuple_out + tuple(map(tuple, arr))

    #return theta_tuple_out
    return theta_tuple_out, y_hat, cost_history #uncomment this line out when using simulated received signal



if __name__ == "__main__": # Change the following code into the c++
    mu, sigma = 0, 0.05
    np.random.seed(0)
    N = 4096   # This also limit the bandwidth. And this is determined by fpga LUT size.
    D = 5
    theta0 = 0.0#+0.1j# *np.random.randn(1, 1) + 0.1j  # parameter to learn
    theta = np.repeat(theta0, D)  # this should be a D * 1 column vector

    theta_imag = np.ndarray.tolist(np.imag(theta))
    theta_tuple = ()
    theta_tuple = array2tuple(theta, D)
    s = np.random.normal(mu, sigma, [N, 1])
    upsamp_rate = 100


    M = 200
    y0 = tx_template(N, M, upsamp_rate)  # rx_sim(N)
    cost_history_all = np.zeros(M)
    for i in range(0,M,1): # this for loop sweeps the received signal fractional time delay
        y = y0[:,i]# - rx_sim(N)
        y_withnoise = y + s
        y = y#y_withnoise

        y_hat = np.zeros([N, 1])
        start = timeit.default_timer()
        for itt in range(10):
            rx_error_sim = y_hat + y  # uncomment this line out when using simulated received signal
            theta_tuple, y_hat, cost_history = main( theta_tuple, N, D, rx_error_sim) # uncomment this line out when using simulated received signal
            #theta_tuple = main(theta_tuple, N, D)
            #print(itt)
        cost_history_all[i] = np.array(cost_history)[0][0]
    import matplotlib.pyplot as plt
    plt.plot(cost_history_all)
    plt.figure()
    plt.plot(rx_error_sim,'*-')
    plt.plot(y_hat,'*-')
    plt.show()
    stop = timeit.default_timer()
    print('Time: ', stop - start)


