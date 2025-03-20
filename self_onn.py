import tensorflow as tf
from tensorflow.keras import layers, regularizers

class SparseAutoencoderNonLinear(tf.keras.Model):
    def __init__(self, n, q, num_conv_layers=2, activation='tanh', lambda_l1=0.01):
        super(SparseAutoencoderNonLinear, self).__init__(name='SparseAutoencoderNonLinear')
        self.n = n
        self.q = q
        self.lambda_l1 = lambda_l1
        self.num_conv_layers = num_conv_layers
        self.activation=activation
        self.diagonal = tf.zeros(n)
        # Dictionary to hold multiple Conv1D layers for different powers
        self.conv_layers = {}
        for degree in range(1, q + 1):  # Powers from x^1 to x^q
            self.conv_layers[degree] = []
            for _ in range(num_conv_layers):  # Multiple Conv1D layers per degree
                self.conv_layers[degree].append(layers.Conv1D(filters=n, kernel_size=3, padding='same', activation='relu' , kernel_initializer=tf.keras.initializers.GlorotNormal()
 ))
        
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
                x_transformed = conv(x_transformed)
                # x_transformed = layers.Activation('swish')(x_transformed) # Apply multiple Conv1D layers
            multi_scale_outputs.append(x_transformed)
        
        # Sum all transformed outputs
        x = tf.add_n(multi_scale_outputs)
        

        
        # Apply L1 regularization in the final layer
        x = self.final_layer(x)
        if self.activation == 'tanh':
            x = tf.nn.tanh(x)
        elif self.activation == 'swish':
            x = tf.nn.swish(x)
        elif self.activation == 'relu':
            x = tf.nn.relu(x)
        elif self.activation=='leakyRelu':
            x = layers.LeakyReLU(alpha=0.01)(x)


        x = tf.vectorized_map(fn=diag_zero, elems = x) # Diagonal constraint.
        # print(f"!!!!!!!!!!!!!!!!!!!Shape is {x.shape}!!!!!!!!!!!!!!!")
        
        return x

# Define input shape (batch_size, n, 1)


# # Compile with Mean Squared Error loss
# model.compile(optimizer='adam', loss='mse')

# # Model summary
# model.build(input_shape=(None, n, 1))
# model.summary()

