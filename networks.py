import tensorflow as tf
import numpy as np

from layers import Oper1D

np.random.seed(10)
tf.random.set_seed(10)

# Ensure that TensorFlow is using the GPU if available
if tf.config.list_physical_devices('GPU'):
    print("GPU is available!")
else:
    print("GPU not found. The model will run on CPU.")

### SLR-OL
def SLRol(n_bands, q):
    with tf.device('/GPU:0'):  # Ensure the model is moved to GPU if available
        input = tf.keras.Input((n_bands, 1), name='input')
        x_0 = Oper1D(n_bands, 3, activation='tanh', q=q)(input)
        y = tf.keras.layers.Dot(axes=(1, 1))([x_0, input])

        model = tf.keras.models.Model(input, y, name='OSEN')

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        model.compile(optimizer=optimizer, loss='mse')

        model.summary()

    return model
