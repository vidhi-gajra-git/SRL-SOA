import tensorflow as tf
import numpy as np
from layers import Oper1D

np.random.seed(10)
tf.random.set_seed(10)

# SLR-OL Model
def SLRol(n_bands, q):
    # Explicitly place the model on the selected GPU
    with tf.device('//CPU:0'):  # You can use '/device:GPU:1', '/device:GPU:2', etc., as needed
        input = tf.keras.Input((n_bands, 1), name='input')
        x_0 = Oper1D(n_bands, 3, activation='tanh', q=q)(input)
        y = tf.keras.layers.Dot(axes=(1, 1))([x_0, input])

        model = tf.keras.models.Model(input, y, name='OSEN')

        # Use GPU for training with Adam optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        # Compile the model and move to GPU
        model.compile(optimizer=optimizer, loss='mse')

        model.summary()

    return model
