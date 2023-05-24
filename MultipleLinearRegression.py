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


# input variables
X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])

# values for w
w_values = np.zeros(len(X_train[0]))
print(compute_gradient(X_train, y_train, w_values, 0.0, 5.0e-8, 10000))

# if this error message "RuntimeWarning: invalid value encountered in scalar subtract" is displayed,
# then that means that your alpha level is too small