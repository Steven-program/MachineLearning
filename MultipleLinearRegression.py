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

mu = np.mean(X_train, axis=0)
sigma = np.std(X_train, axis=0)
X_norm = (X_train - mu) / sigma
# the training data is now normalized

# values for w
w_values = np.zeros(len(X_train[0]))
w, b = compute_gradient(X_norm, y_train, w_values, 0.0, 1.0e-1, 1000)
print(w, b)

# if this error message "RuntimeWarning: invalid value encountered in scalar subtract" is displayed,
# then that means that your alpha level is too small or too big

# when the training data is normalized, further training data must also be normalized. Example:
x_house = np.array([1200, 3, 1, 40])
mu = np.mean(x_house)
sigma = np.std(x_house)
x_house = (x_house - mu)/sigma
prediction = np.dot(x_house, w) + b
print(prediction)