#  https://towardsdatascience.com/gradient-descent-in-python-a0d07285742f
import numpy as np
import matplotlib.pyplot as plt

def cal_cost(theta, X, y):
    m = len(y)
    predictions = X.dot(theta)
    A = predictions - y
    cost = 1.0 / (2 * m) * np.dot(np.conj(A).T, A)[0][0]
    return cost


def gradient_descent(X, y, theta, learning_rate=0.01, iterations=100):
    m = len(y)
    cost_history = np.zeros(iterations)
    theta_history = np.zeros((iterations, len(theta)))
    for it in range(iterations):
        prediction = np.dot((X), theta)
        error = prediction - y
        theta = theta - (2.0 / m) * learning_rate * np.dot(np.conj(X.T),error ) # change to CPP
        #theta = theta - (1.0 / m) * learning_rate * np.dot(np.conj(X.T), error )  # change to CPP
        theta_history[it, :] = theta.T
        cost_history[it] = cal_cost(theta, X, y)
    return  theta, cost_history, theta_history


def stocashtic_gradient_descent(X, y, theta, learning_rate=0.01, iterations=10):
    '''
    X    = Matrix of X with added bias units
    y    = Vector of Y
    theta=Vector of thetas np.random.randn(j,1)
    learning_rate
    iterations = no of iterations

    Returns the final theta vector and array of cost history over no of iterations
    '''
    m = len(y)
    cost_history = np.zeros(iterations)

    for it in range(iterations):
        cost = 0.0
        for i in range(m):
            rand_ind = np.random.randint(0, m)
            X_i = X#[rand_ind, :].reshape(1, X.shape[1])
            y_i = y[rand_ind].reshape(1, 1)
            prediction = np.dot(X_i, theta)

            theta = theta - (1 / m) * learning_rate * np.dot(np.conj(X_i.T), prediction - y_i )
                #(X_i.T.dot((prediction - y_i)))
            cost += cal_cost(theta, X_i, y_i)
        cost_history[it] = cost

    return theta, cost_history
