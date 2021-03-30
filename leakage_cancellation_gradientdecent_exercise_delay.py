import numpy as np
import matplotlib.pyplot as plt


def cal_cost(theta, X, y):
    m = len(y)
    y_prediction = X.dot(theta)
    A = y_prediction - y
    cost = 1.0 / (2 * m) * np.dot(np.conj(A).T, A)[0][0]
    return cost


def gradient_descent(X, y, theta, learning_rate=0.01, iterations=100):
    N = len(y)
    cost_history = np.zeros(iterations)
    theta_history = np.zeros((iterations, len(theta)))
    prediction = np.zeros([N,1])
    mu, sigma = 0, 0.01
    att = 200

    for it in range(iterations):
        s = np.random.normal(mu,sigma,[N,1])
        y_withnoise = y + s
        print(theta)
        error = att *(prediction -  y)#_withnoise # after combination of y and x_canc, there could be attenuation and delay due to T.L and FPGA.
        #error = np.concatenate((error[N - 800:], error[:N - 800]), axis=0) # this can make convergence slower
        theta = theta - (1.0 / N) * learning_rate *np.dot(np.conj(X.T),error/att ) # change to CPP
        prediction = np.dot((X), theta)
        prediction = np.concatenate((prediction[800 : N ], prediction[0 : 800]), axis=0) # compensate the the auxilary transmit path delay
        # external phase shift and attenuation
        prediction = np.concatenate((prediction[N - 800:], prediction[:N - 800]), axis=0)
        prediction =  prediction

        theta_history = np.zeros([iterations,theta.shape[0],theta.shape[1]]) # create initial matrix storing theta which is D by 1
        theta_history[it] = theta
        cost_history[it] = cal_cost(theta, X, y)[0,0]

        x_canc = X.dot(theta)
        '''
        ############
        # Plot
        ############
        N=m
        fs = 28e6  # Sampling freq
        tc = N / fs  # T=N/fs#Chirp Duration
        t = np.linspace(0, tc, N)
        plt.figure()
        x=X
        plt.plot(t, x, 'b-', t, y, 'k-', t, x_canc, 'r*')
        plt.legend(['x', 'y', 'x_canc'])
        plt.show()
        '''
    return  theta, cost_history, theta_history





def main():
    #############
    # Parameters
    #############
    j = 1j
    fs = 28e6  # Sampling freq
    N = 4096  # This also limit the bandwidth. And this is determined by fpga LUT size.
    tc = N/fs  # T=N/fs#Chirp Duration
    t = np.linspace(0, tc, N)
    f0 = -14e6  # Start Freq
    f1 = 14e6  # fs/2=1/2*N/T#End freq
    K = (f1 - f0) / tc # chirp rate = BW/Druation
    #############
    # Waveform
    #############
    #x0 = 1.0 * np.sin(2 * np.pi * fs / N * t)
    #xq0 = 1.0 * np.sin(2 * np.pi * fs / N * t - np.pi / 2)
    x0 = 1*np.sin(2 * np.pi * (f0 * t + K / 2 * np.power(t, 2)))  # use this for chirp generation
    xq0 = 1*np.sin(2 * np.pi * (f0 * t + K / 2 * np.power(t, 2))- np.pi / 2)  # use this for chirp generation
    x = x0 + j*xq0

    D = 100 # Dimontion of X matrix.Total_delay_tap
    x_cx_delay = j*np.ones([N,D])
    for i in range (D):
        x_cx_delay[:,i] = np.roll(x, 200+i)
    x = np.matrix(x_cx_delay)
    '''
    x_cx_delay_0 = np.concatenate((x[N-200:], x[:N - 200]),axis=0)
    x_cx_delay_1 = np.concatenate((x[N - 200-1:], x[:N - 200-1]), axis=0)
    x_cx_delay_2 = np.concatenate((x[N - 200 - 2:], x[:N - 200 - 2]), axis=0)
    x = np.matrix(np.transpose(
    [x_cx_delay_0, x_cx_delay_1, x_cx_delay_2]))#x_cx_delay
    '''

    Amp = 0.2
    phi_init1 = -10.50/180.0*np.pi
    phi_init2 = -10.50/180.0*np.pi#-105.0/180.0*np.pi
    dc_offset = 0

    #y0 = Amp * np.sin(2 * np.pi * fs / N * t + phi_init1) #+ dc_offset # + numpy.sin(4*numpy.pi*fs/N*t)# just use LO to generate a LO. The
    #yq0 = Amp * np.sin(2 * np.pi * fs / N * t - np.pi / 2 + phi_init2) + dc_offset  # + numpy.sin(4*numpy.pi*fs/N*t-numpy.pi/2)
    y0 = Amp*np.sin(phi_init1 + 2 * np.pi * (f0 * t + K / 2 * np.power(t, 2))) + dc_offset # use this for chirp generation
    yq0 = Amp*np.sin(phi_init2 + 2 * np.pi * (f0 * t + K / 2 * np.power(t, 2))- np.pi / 2) + dc_offset # use this for chirp generation
    y = y0 + j * yq0
    y_cx_delay = np.concatenate((y[N-200-(D-1):], y[:N - 200-(D-1)]),axis=0)
    y = y_cx_delay
    #################
    # Gradient Decent
    #################
    np.random.seed(100)
    theta0 = np.array([0.1+0.1j])#np.random.randn(1,1)+0.1j # cannot use zeros for theta initial value, wont converge.
    theta = np.matrix(np.repeat(theta0, x.shape[1])).T # this should be a D * 1 column vector
    lr = 1
    n_iter = 300

    X = x#x[np.newaxis].T
    Y = y[:,np.newaxis] # N by 1 column vector

    theta, cost_history, theta_history = gradient_descent(X, Y, theta, lr, n_iter)

    x_canc = X.dot(theta)
    ############
    # Plot
    ############

    fig,ax = plt.subplots(figsize=(12,8))
    ax.set_ylabel('J(Theta)')
    ax.set_xlabel('Iterations')
    ax.plot(range(n_iter),10*np.log10(cost_history),'b.')

    plt.figure()
    plt.plot(t,x[:,0],'b-',t,y,'k-',t,x_canc,'r*')
    plt.legend(['x','y','x_canc'])
    plt.show()


if __name__ == "__main__":
    main()
