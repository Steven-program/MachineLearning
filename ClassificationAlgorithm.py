import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


def sigmoid(num):
    return 1 / (1 + np.exp(-num))


def calculate_DLoss(x, y, w, b, lambd = 1):
    m, n = x.shape
    dw = np.zeros(len(w))
    db = 0.0

    for i in range(m):
        z = sigmoid(np.dot(w, x[i]) + b)
        loss = z - y[i]
        for j in range(n):
            dw[j] += x[i, j] * loss
        db += loss

    dw /= m
    db /= m

    for j in range(n):
        dw[j] += lambd/m * w[j]
    return dw, db


def gradient_descent(x, y, w, b, alpha, iterations):
    for i in range(iterations):
        dw, db = calculate_DLoss(x, y, w, b)
        for s in range(len(w)):
            w[s] -= alpha * dw[s]
        b -= alpha * db

    return w, b


def alternative_method(x, y):
    model = LogisticRegression()
    model.fit(x, y)
    y_p = model.predict(x)
    print(y_p)


X_train = np.array([[0.5, 1.5], [1, 1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])  # (m,n)
y_train = np.array([0, 0, 0, 1, 1, 1])

w_input = np.zeros(len(X_train[0]))
w,b = gradient_descent(X_train, y_train, w_input, b=0.0, alpha=0.1, iterations=10000)
print(w, b)

predictions = np.zeros(len(X_train))
for i in range(len(X_train)):
    predictions[i] = sigmoid(np.dot(X_train[i], w) + b)
print(predictions)

alternative_method(X_train, y_train)