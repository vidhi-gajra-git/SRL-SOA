import tensorflow as tf
from tensorflow.keras import layers, regularizers

class SparseAutoencoderNonLinear2(tf.keras.Model):
    def __init__(self, n, q, kernel_size, num_conv_layers=2, activation='tanh', lambda_l1=0.01):
        super(SparseAutoencoderNonLinear2, self).__init__(name=f'SparseAutoencoder_k{kernel_size}')
        self.n = n
        self.q = q
        self.kernel_size = kernel_size
        self.lambda_l1 = lambda_l1
        self.num_conv_layers = num_conv_layers
        self.activation = activation
        self.diagonal = tf.zeros(n)

        # Dictionary to hold multiple Conv1D layers for different powers
        self.conv_layers = {}
        for degree in range(1, q + 1):  # Powers from x^1 to x^q
            self.conv_layers[degree] = []
            for _ in range(num_conv_layers):  # Multiple Conv1D layers per degree
                self.conv_layers[degree].append(layers.Conv1D(filters=n, kernel_size=kernel_size, padding='same', activation=activation))
        
        # Final Conv1D layer with L1 regularization
        self.final_layer = layers.Conv1D(
            filters=n, kernel_size=1, padding='same', activation=None,  
            kernel_regularizer=regularizers.l1(self.lambda_l1)
        )
    def get_hyperparameters(self):
        return {k: v for k, v in self.__dict__.items() if isinstance(v, (int, float, str, bool))}

    def call(self, inputs):
        multi_scale_outputs = []
        def diag_zero(input):
            x_0 = tf.linalg.set_diag(input, self.diagonal)
            return x_0
        
        # Apply different non-linear transformations with multiple Conv1D layers
        for degree, conv_list in self.conv_layers.items():
            x_transformed = tf.math.pow(inputs, degree)  # Apply non-linearity (x^degree)
            for conv in conv_list:
                x_transformed = conv(x_transformed)  # Apply multiple Conv1D layers
            multi_scale_outputs.append(x_transformed)
        
        # Sum all transformed outputs
        x = tf.add_n(multi_scale_outputs)

        # Apply L1 regularization in the final layer
        x = self.final_layer(x)
        if self.activation is not None:
            x = eval('tf.nn.' + self.activation + '(x)')

        x = tf.vectorized_map(fn=diag_zero, elems=x) 
        # Diagonal constraint
        
        return x


class MultiKernelEncoder(tf.keras.Model):
    def __init__(self, n, q, num_conv_layers=2, activation='tanh', lambda_l1=0.01, final_op='multiply'):
        super(MultiKernelEncoder, self).__init__(name='MultiKernelEncoder')
        self.encoder_3 = SparseAutoencoderNonLinear2(n, q, kernel_size=3, num_conv_layers=num_conv_layers, activation=activation, lambda_l1=lambda_l1)
        self.encoder_5 = SparseAutoencoderNonLinear2(n, q, kernel_size=5, num_conv_layers=num_conv_layers, activation=activation, lambda_l1=lambda_l1)
        self.encoder_7 = SparseAutoencoderNonLinear2(n, q, kernel_size=7, num_conv_layers=num_conv_layers, activation=activation, lambda_l1=lambda_l1)

        self.final_op = final_op
        self.diagonal = tf.zeros(n)

        if final_op == 'conv':
            self.final_conv = layers.Conv2D(filters=1, kernel_size=(3,3), padding='same', activation='tanh')
    def diag_zero(input):
            x_0 = tf.linalg.set_diag(input, self.diagonal)
            return x_0
    def get_hyperparameters(self):
        dictt=encoder_3.get_hyperparameters()
        dictt['architecture']=[3,5,7]
        
        return dictt
    def call(self, inputs):
        op1 = self.encoder_3(inputs)
        op2 = self.encoder_5(inputs)
        op3 = self.encoder_7(inputs)

        if self.final_op == 'multiply':
            output = op1 * op2 * op3  # Element-wise multiplication
        elif self.final_op == 'conv':
            # Stacking along channel dimension to perform 2D conv
            x = tf.stack([op1, op2, op3], axis=-1)  # Shape (batch_size, n, n, 3)
            output = self.final_conv(x)
            x = tf.vectorized_map(fn=diag_zero, elems=output) 
            output=x
        else:
            raise ValueError("Invalid final_op. Choose 'multiply' or 'conv'.")

        return output

# Example usage:
# n = 64  # Example size
# q = 3   # Polynomial degree
# batch_size = 8

# # Define input shape (batch_size, n, 1)
# inputs = tf.keras.Input(shape=(n, 1))
# model = MultiKernelEncoder(n=n, q=q, final_op='conv')
# output = model(inputs)

# # Compile with Mean Squared Error loss
# model.compile(optimizer='adam', loss='mse')

# # Model summary
# model.build(input_shape=(None, n, 1))
# model.summary()
