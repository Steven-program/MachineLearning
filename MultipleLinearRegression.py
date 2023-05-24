import numpy as np

# input variables
X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])

def compute_derivative(x, y, w, b):
    dw, db = 0, 0
    total_diff, total_diff_b = 0, 0

    m = len(x)
    for i in range(m):
        diff = np.dot(w, x + b - y[i]
        total_diff += (diff * x[i])
        total_diff_b += diff

    dw = total_diff / m
    db = total_diff_b / m

    return dw, db


