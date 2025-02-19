import tensorflow as tf
from tensorflow.keras import regularizers

class Oper1DDilated(tf.keras.Model):
    def __init__(self, filters, dilation_rates=[1, 2, 4], activation='relu', q=1):
        super(Oper1DDilated, self).__init__(name='')
        self.diagonal = tf.zeros(filters)
        self.activation = activation
        self.q = q
        self.lambda_ = 0.001
        self.all_layers = {}

        # Create dilated convolution layers
        self.dilated_convs = []
        for rate in dilation_rates:
            self.dilated_convs.append(
                tf.keras.layers.Conv1D(
                    filters, kernel_size=3, padding='same',
                    dilation_rate=rate, activation='relu',
                    kernel_regularizer=regularizers.l1(self.lambda_)
                )
            )

        # Final Combining Layer
        self.combine_layer = tf.keras.layers.Conv1D(filters, kernel_size=1, padding='same', activation='relu')

    @tf.function
    def call(self, input_tensor, training=False):
        def diag_zero(x):
            return tf.linalg.set_diag(x, self.diagonal)

        dilated_outputs = []
        for conv in self.dilated_convs:
            x_dilated = conv(input_tensor)
            dilated_outputs.append(x_dilated)

        # Concatenating dilated convolution outputs
        x = tf.concat(dilated_outputs, axis=-1)

        # Regularization
        x = tf.keras.layers.ActivityRegularization(l1=0.001)(x) 

        # Final Combining Layer
        x = self.combine_layer(x)

        if self.activation is not None:
            x = eval('tf.nn.' + self.activation + '(x)')

        x = tf.vectorized_map(fn=diag_zero, elems=x)

        return x
