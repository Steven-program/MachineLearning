import numpy as np
import matplotlib as plt
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense

def compute_decisions(predict):
    yhat = np.zeros_like(predict)
    for i in range(len(predict)):
        if predict[i] >= 0.5:
            yhat[i] = 1
        else:
            yhat[i] = 0

    return yhat

x = np.array([[200.0, 17.0],
              [120.0, 5.0],
              [425.0, 20.0],
              [212.0, 18.0]])
y = np.array([1, 0, 0, 1])

# first let's normalize our data to increase its efficiency
norm_1 = tf.keras.layers.Normalization(axis=-1)
norm_1.adapt(x)
xn = norm_1(x)

y.shape = (4, 1)

# then we "tile"/copy our data to increase the training set size and reduce the number of training epochs (reduces
# overfitting)
xt = np.tile(xn, (1000,1))
yt = np.tile(y, (1000, 1))
print(xt.shape, yt.shape)
# this sets up a neural network with two layers
tf.random.set_seed(1234)
model = Sequential([
    Dense(units=3, activation="sigmoid", name="layer_1"),
    Dense(units=1, activation="sigmoid", name="layer_2")
])

model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01)
)
model.fit(xt, yt, epochs=10)

W1, b1 = model.get_layer("layer_1").get_weights()
W2, b2 = model.get_layer("layer_2").get_weights()

X_test = np.array([
    [200, 13.9],
    [200, 17]
])
X_test_norm = norm_1(X_test)
predictions = model.predict(X_test_norm)
decision = compute_decisions(predictions)

print(decision)