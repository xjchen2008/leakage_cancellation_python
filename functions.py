import numpy as np

def equalizer(x, y):
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
    return x_EQ