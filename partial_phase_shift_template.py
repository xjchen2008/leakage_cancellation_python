import numpy as np
import matplotlib.pyplot as plt
import readbin
import gradient_descent_v3 as gd
def main():
    #############
    # Parameters
    #############
    c = 3e8
    j = 1j
    fs = 28e6  # Sampling freq
    N = 4096+1  # This also limit the bandwidth. And this is determined by fpga LUT size.
    tc = N/fs  # T=N/fs#Chirp Duration
    t = np.linspace(0, tc, N)
    nn = np.linspace(0, N-1, N)
    f0 = -14e6  # Start Freq
    f1 = 14e6  # fs/2=1/2*N/T#End freq
    K = (f1 - f0) / tc # chirp rate = BW/Druation
    NFFT =  N
    freq = np.fft.fftfreq(NFFT, d=1 / fs)
    distance = c * freq / K / 2.0
    phi0 = -(4999+1) * np.pi / 10000  # Phase
    win=np.blackman(N)
    #win=np.hamming(N)
    win=1

    # phase shifted signal
    part = 1000
    tx_sig_sim = j * np.ones((N,part))
    for i in range(0, part, 1):

        phi_init = 2 * np.pi / N / (part-1) * i

        y0 = 1*np.sin(2 * np.pi * (f0 * t + K / 2 * np.power(t, 2)) + phi_init) + 0.0 # use this for chirp generation
        yq0 = 1*np.sin(phi0 + 2 * np.pi * (f0 * t + K / 2 * np.power(t, 2)) + phi_init)  # use this for chirp generation

        #y0 = 1*np.sin(2 * np.pi * fs / N * t + phi_init)  # + numpy.sin(4*numpy.pi*fs/N*t)# just use LO to generate a LO. The
        #yq0 = 1*np.sin(2 * np.pi * fs / N * t - np.pi / 2 + phi_init)  # + numpy.sin(4*numpy.pi*fs/N*t-numpy.pi/2)

        tx_sig_sim[:,i] = y0 + j * yq0
    print('end')

    #################
    # Plot
    #################
    #plt.figure()
    #plt.plot(tx_sig_sim,'*-')
    #lt.title('Template Signal in Time Domain')
    #plt.grid()
    #plt.show()
if __name__ == "__main__":
    main()
