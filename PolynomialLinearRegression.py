import numpy as np

np.set_printoptions(precision=2)


def compute_derivative(x, y, w, b):
    dw = np.zeros(len(w))
    db = 0
    m, n = x.shape
    # m = rows, n = columns
    for i in range(m):
        diff = np.dot(x[i], w) + b - y[i]
        for j in range(n):
            dw[j] += (diff * x[i, j])
        db += diff

    dw *= 1 / m
    db *= 1 / n

    return dw, db


def compute_gradient(x, y, w, b, alpha, iterations):
    for iter in range(iterations):
        dw, db = compute_derivative(x, y, w, b)
        for i in range(len(w)):
            w[i] -= alpha * dw[i]
            # print(w[i])
        b -= alpha * db
    return w, b


x = np.arange(0, 20, 1)
y = 1 + x ** 2
X = x ** 2
X = X.reshape(-1, 1)
# print(X)
w_values = np.zeros(len(X[0]))
print(compute_gradient(X, y, w_values, b=0, iterations=40000, alpha=1e-5))
