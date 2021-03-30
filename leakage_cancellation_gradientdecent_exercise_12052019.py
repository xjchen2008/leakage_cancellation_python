import numpy as np
import matplotlib.pyplot as plt


def cal_cost(theta, X, y):
    m = len(y)
    y_prediction = X.dot(theta)
    A = y_prediction - y
    cost = 1.0 / (2 * m) * np.dot(np.conj(A).T, A)[0][0]
    return cost


def gradient_descent(X, y, theta, learning_rate=0.01, iterations=100):
    m = len(y)
    cost_history = np.zeros(iterations)
    theta_history = np.zeros((iterations, len(theta)))
    for it in range(iterations):
        print(theta)
        prediction = np.dot((X), theta)
        error = prediction - y
        theta = theta - (1.0 / m) * learning_rate * np.dot(np.conj(X.T),error ) # change to CPP

        theta_history[it, :] = theta.T
        cost_history[it] = cal_cost(theta, X, y)
    return  theta, cost_history, theta_history


def main():
    #############
    # Parameters
    #############
    j = 1j
    fs = 28e6  # Sampling freq
    N = 4096+1  # This also limit the bandwidth. And this is determined by fpga LUT size.
    tc = N/fs  # T=N/fs#Chirp Duration
    t = np.linspace(0, tc, N)
    f0 = -14e6  # Start Freq
    f1 = 14e6  # fs/2=1/2*N/T#End freq
    K = (f1 - f0) / tc # chirp rate = BW/Druation
    #############
    # Waveform
    #############
    x0 = 1.0 * np.sin(2 * np.pi * fs / N * t)
    xq0 = 1.0 * np.sin(2 * np.pi * fs / N * t - np.pi / 2)
    #x0 = 1*np.sin(2 * np.pi * (f0 * t + K / 2 * np.power(t, 2)))  # use this for chirp generation
    #xq0 = 1*np.sin(2 * np.pi * (f0 * t + K / 2 * np.power(t, 2)))  # use this for chirp generation
    x = x0 + j*xq0

    Amp = 2
    phi_init1 = -105.0/180.0*np.pi
    phi_init2 = -105.0/180.0*np.pi
    dc_offset = 0

    y0 = Amp * np.sin(2 * np.pi * fs / N * t + phi_init1) + dc_offset # + numpy.sin(4*numpy.pi*fs/N*t)# just use LO to generate a LO. The
    yq0 = Amp * np.sin(2 * np.pi * fs / N * t - np.pi / 2 + phi_init2) + dc_offset  # + numpy.sin(4*numpy.pi*fs/N*t-numpy.pi/2)
    #y0 = Amp*np.sin(phi_init1 + 2 * np.pi * (f0 * t + K / 2 * np.power(t, 2))) + dc_offset # use this for chirp generation
    #yq0 = Amp*np.sin(phi_init2 + 2 * np.pi * (f0 * t + K / 2 * np.power(t, 2))) + dc_offset # use this for chirp generation
    y = y0 + j * yq0
    y_cx_delay = np.concatenate((y[N-200:], y[:N - 200]),axis=0)
    y = y#y_cx_delay
    #################
    # Gradient Decent
    #################
    np.random.seed(100)
    theta = np.random.randn(1,1) # cannot use zeros for theta initial value, wont converge.
    lr = 0.1
    n_iter = 4000

    X = x[np.newaxis].T
    Y = y[np.newaxis].T
    theta, cost_history, theta_history = gradient_descent(X, Y, theta, lr, n_iter)
    print(theta)
    x_canc = X.dot(theta)
    ############
    # Plot
    ############

    fig,ax = plt.subplots(figsize=(12,8))
    ax.set_ylabel('J(Theta)')
    ax.set_xlabel('Iterations')
    ax.plot(range(n_iter),10*np.log10(cost_history),'b.')

    plt.figure()
    plt.plot(t,x,'b-',t,y,'k-',t,x_canc,'r*')
    plt.legend(['x','y','x_canc'])
    plt.show()


if __name__ == "__main__":
    main()
