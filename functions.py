import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fftshift
import dsp_filters

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


def equalizer(x, y, input):
    #####################
    # Equalization filter:
    # Equalizers are used to render the frequency response and flat it from end-to-end
    # x is the ideal signal. y is the received signal.

    #####################
    #X = np.fft.fft(x)
    #Y = np.fft.fft(y)
    y = y /max(abs(y))
    X = np.fft.rfft(x) # https://stackoverflow.com/questions/52387673/what-is-the-difference-between-numpy-fft-fft-and-numpy-fft-rfft
    Y = np.fft.rfft(y)
    INPUT = np.fft.rfft(input)
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
    x_EQ = np.fft.irfft(X_EQ)
    x_EQ = x_EQ/max(abs(x_EQ))
    #
    INPUT_EQ = np.multiply(H_inv, INPUT)
    INPUT_EQ[0] = 0 # setting the bin zero(DC component to zero) to get rid of DC offset. It is not zero due to noise and randomness.
    input_EQ = np.fft.irfft(INPUT_EQ)
    input_EQ = input_EQ/max(abs(input_EQ))
    return input_EQ


def PulseCompr(rx,tx,win, unit = 'log'):
    # Mixer method pulse compression; Return a log scale beat frequency signal.
    a = np.multiply(rx,win)  #np.power(win, 10)#np.multiply(win,win) # Add window here
    b = np.multiply(tx,win)  #np.power(win, 10)#np.multiply(win,win)#tx
    mix = b * np.conj(a)  # 1. time domain element wise multiplication.
    pc = np.fft.fft(mix)  # 2. Fourier transform.
    pc_timedomain = np.fft.ifft(pc)
    pc_LPF = dsp_filters.main(signal=pc_timedomain, order=6, fs=250e6, cutoff=6e6, duration=1.6e-6)
    pc_LPF_freqdomain = np.fft.fft(pc_LPF)
    # match filter method
    A = np.fft.fft(a)
    B = np.fft.fft(b)
    pc_mf = np.fft.ifft(np.multiply(B, np.conj(A)))
    if unit == 'log':
        pc = 20 * np.log10(abs(pc_LPF_freqdomain))
    if unit =='linear':
        pc = pc
    return pc


def channel_est(psi_orth, y_cx_received):
    # Calculate weights
    # 1. Make H matrix
    N= len(y_cx_received)
    H = psi_orth
    D = 100
    for idx in range(1, D):
        psi_orth_delay_roll = np.roll(psi_orth, idx)
        H = np.column_stack((H, psi_orth_delay_roll))
        #print(H)
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


def plot_freq_db(freq, x, color='b'):
    # This function plots the input in frequency domain in dB
    X = np.fft.fft(x,axis = 0)
    #freq = np.fft.fftfreq(len(x))
    X_log = 20*np.log10(np.abs(X))
    X_log_normalize = X_log - max(X_log)
    plt.plot(fftshift(freq), fftshift(X_log_normalize), color = color)
    plt.grid(b= True)
    return X


def normalize(x):
    x_abs = abs(x)
    y = x - max(x_abs)
    return y