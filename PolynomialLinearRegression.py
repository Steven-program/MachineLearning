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


x = np.arange(0,20,1)
y = np.cos(x/2)
X = np.c_[x, x**2, x**3, x**4, x**5, x**6, x**7, x**8, x**9, x**10, x**11, x**12, x**13]

# feature scaling to improve time and cut alpha level
mu = np.mean(X, axis=0)
sigma = np.std(X, axis=0)
X_norm = (X - mu) / sigma

w_values = np.zeros(len(X[0]))
print(compute_gradient(X_norm, y, w_values, b=0, iterations=1000000, alpha=1e-1))
