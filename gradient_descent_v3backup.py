#  https://towardsdatascience.com/gradient-descent-in-python-a0d07285742f
import numpy as np


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
        theta = theta - (1.0 / m) * learning_rate * np.dot(error.T, np.conj(X) ) # change to CPP
        theta_history[it, :] = theta.T
        cost_history[it] = cal_cost(theta, X, y)
    return  theta, cost_history, theta_history
