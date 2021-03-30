#  https://towardsdatascience.com/gradient-descent-in-python-a0d07285742f

import numpy as np


def cal_cost(theta, X, y):

    m = len(y)
    predictions = np.dot(X.T, theta)
    A = predictions - y
    #cost = 1.0 /( 2 * m) * np.sum(np.dot(np.conj(A).T, A )) #
    cost = 1.0 / (2 * m) * np.dot(np.conj(A).T, A)[0][0]
    return cost


def gradient_descent(X, y, theta, learning_rate=0.01, iterations=100, epsilon = 1e-3):
    m = len(y)
    cost_history = np.zeros(iterations)
    theta_history = np.zeros((iterations, len(theta)))
    for it in range(iterations):
        prediction = np.dot((X.T), theta)

        theta = theta - (1.0 / m) * learning_rate * np.dot(np.conj(X), prediction - y)
        theta_history[it, :] = theta.T

    return theta, cost_history, theta_history