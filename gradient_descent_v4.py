#  https://towardsdatascience.com/gradient-descent-in-python-a0d07285742f
import numpy as np
import matplotlib.pyplot as plt

def cal_cost(theta, X, y):
    m = len(y)
    y_prediction = X.dot(theta)
    A = y_prediction - y
    cost = 1.0 / (2 * m) * np.dot(np.conj(A).T, A)[0][0]
    return cost


def gradient_descent(X, error, theta, y0, learning_rate=0.01, iterations=100):
    m = len(error)
    cost_history = np.zeros(iterations)
    theta_history = np.zeros((iterations, len(theta)))
    y = y0
    for it in range(iterations):
        theta = theta - (2.0 / m) * learning_rate * np.dot(np.conj(X.T), error )  # change to CPP
        y_prediction = np.dot((X), theta)
        error = y_prediction - y
        theta_history[it, :] = theta.T
        cost_history[it] = cal_cost(theta, X, y)
    return  theta, cost_history, theta_history
