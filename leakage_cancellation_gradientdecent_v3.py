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
    phi_init = 0.5
    y0 = 1*np.sin(2 * np.pi * (f0 * t + K / 2 * np.power(t, 2))) + 0.0 # use this for chirp generation
    yq0 = 1*np.sin(phi0 + 2 * np.pi * (f0 * t + K / 2 * np.power(t, 2)))  # use this for chirp generation
    y0 = -0.1*np.sin(2 * np.pi * fs / N * t + phi_init)  # + numpy.sin(4*numpy.pi*fs/N*t)# just use LO to generate a LO. The
    yq0 = -0.1*np.sin(2 * np.pi * fs / N * t - np.pi / 2 + phi_init)  # + numpy.sin(4*numpy.pi*fs/N*t-numpy.pi/2)

    tx_sig_sim = y0 + j * yq0
    ###################
    # Import Signals
    ###################
    # Transmited signal
    outfile = "ref_signal_20191118_avg1_28MHz.npy"
    tx_sig = np.load(outfile)
    # Received signal
    nsamples = N
    times =1
    rubish = 4096*90
    num_of_channel = 2
    chirp_length = N
    rx_data = readbin.average("usrp_samples_loopback.dat", nsamples, times, rubish, num_of_channel, chirp_length)
    rx_ch0, rx_ch1 = rx_data
    rx_ch0 = np.array(rx_ch0[0:N])
    rx_ch1 = np.array(rx_ch1[0:N])
    rx_avg = rx_ch0
    #rx_avg = rx_avg[np.newaxis]

    ######################################
    # Pulse Compression Stretching Method
    #####################################
    def PulseCompr(rx,tx,win):
        A = np.multiply(rx,win) # Add window here
        B = tx
        PC = 20*np.log10(abs(np.fft.fft(A*np.conj(B))))
        return np.squeeze(PC)

    ########################
    # Gradient Decent
    ########################
    # Template
    psi_orth = tx_sig
    psi_orth_delay_0 = np.concatenate((np.zeros(0), psi_orth[:N - 0]), axis=0)
    #psi_orth_delay_1 = np.concatenate((np.zeros(1), psi_orth[:N - 1]), axis=0)
    H = np.matrix(np.transpose([psi_orth_delay_0]))
    np.random.seed(100)
    theta = np.random.randn(H.shape[1],1)
    lr = 0.1
    n_iter = 40000
    const = 1
    X = tx_sig_sim#tx_sig_sim#tx_sig#rx_avg
    y = rx_avg#tx_sig#rx_avg#tx_sig#rx_avg-tx_sig
    X = X[np.newaxis].T
    y = y[np.newaxis].T

    theta,cost_history,theta_history = gd.gradient_descent(X,y,theta,lr,n_iter)
    #theta, cost_history = gd.stocashtic_gradient_descent(X, y, theta, lr, n_iter)

    x_canc_gd = np.squeeze(np.array(X.dot(theta))) # as an array not matrix
    print(theta)

    fig,ax = plt.subplots(figsize=(12,8))
    ax.set_ylabel('J(Theta)')
    ax.set_xlabel('Iterations')
    ax.plot(range(n_iter),10*np.log10(cost_history),'b.')

    #################
    # Plot
    #################
    plt.figure()
    plt.plot(X)
    plt.title('Template Signal in Time Domain')
    plt.figure()
    plt.plot(rx_avg)
    plt.title('Received Signal in Time Domain')
    plt.figure()
    plt.plot(x_canc_gd)
    plt.title('Cancellation Signal in Time Domain')

    y_canc = rx_avg-x_canc_gd
    plt.figure()
    plt.plot(y_canc)
    plt.title('Remaining received signal after cancellation ')
    # Pulse compression after cancellation
    PC = PulseCompr(rx=y_canc,tx=tx_sig,win=np.blackman(N))
    plt.figure()
    plt.plot(distance,PC,'k*-',label='After Leakage Cancellation')
    #plt.xlim((-10,800))
    plt.xlabel('Distance in meter')
    plt.ylabel('Power in dB')
    plt.grid()

    # Pulse compression before cancellation
    PC = PulseCompr(rx = rx_avg,tx = tx_sig,win = np.blackman(N))
    plt.plot(distance,PC,'b-.',label='Before Leakage Cancellation')
    plt.title('Pulse Compression')
    plt.legend()


    # save updated x_canc
    x_canc=np.zeros(2*N)
    x_canc_gd_update = x_canc_gd / max(abs(x_canc_gd)) * 32767*8/10
    for i in range(0, N):
        x_canc[2 * i + 1] = np.imag(x_canc_gd_update[i])  # tx signal
        x_canc[2 * i] = np.real(x_canc_gd_update[i])  # tx signal
    x_canc = np.int16(x_canc)  # E312 setting --type short
    print(max(x_canc))
    data = open('usrp_samples.dat', 'w')
    data.write(x_canc)
    data.close()

    plt.show()
if __name__ == "__main__":
    main()