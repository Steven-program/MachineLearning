import numpy as np


# implement the derivatives of the gradient descent algorithm
def compute_gradient(x, y, w, b):
    dw, db = 0, 0
    total_diff, total_diff_b = 0, 0

    m = len(x)
    for i in range(m):
        diff = w * x[i] + b - y[i]
        total_diff += (diff * x[i])
        total_diff_b += diff

    dw = total_diff / m
    db = total_diff_b / m

    return dw, db


# do the actual gradient descent algorithm
def gradient_descent(w, b, alpha, x, y, iteration):
    for i in range(iteration):
        dw, db = compute_gradient(x, y, w, b)
        temp_w = w - alpha * dw
        temp_b = b - alpha * db
        w = temp_w
        b = temp_b

    return w, b


# Load in training set
x_train = np.array([1.0, 2.0])  # features
y_train = np.array([300.0, 500.0])  # target value

itera = 10000
print(gradient_descent(0, 0, 0.01, x_train, y_train, itera))
