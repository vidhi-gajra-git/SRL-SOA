import tensorflow as tf
from tensorflow.keras import regularizers

class Oper1DMultiScaleDeep(tf.keras.Model):
    def __init__(self, filters, kernel_sizes, activation='relu', q=1, num_layers=3):
        """
        Parameters:
          - filters: Number of filters for each Conv1D per scale.
          - kernel_sizes: List of kernel sizes, e.g., [3, 5, 7].
          - activation: Activation function to apply.
          - q: Number of convolutional layers (order) per scale.
          - num_layers: Number of additional Conv1D layers to add sequentially.
        """
        super(Oper1DMultiScaleDeep, self).__init__(name='')
        self.diagonal = tf.zeros(filters)
        self.activation = activation
        self.q = q
        self.kernel_sizes = kernel_sizes
        self.lambda_ = 0.001

        # Create multi-scale Conv1D layers for each kernel size.
        self.all_layers = {}
        for k_size in kernel_sizes:
            layers_for_scale = []
            for i in range(q):
                layers_for_scale.append(
                    tf.keras.layers.Conv1D(filters, k_size, padding='same', activation='relu',
                                             kernel_regularizer=regularizers.l2(self.lambda_))
                )
            self.all_layers[k_size] = layers_for_scale
        
        # 1x1 convolution layer to combine multi-scale features.
        self.combine_layer = tf.keras.layers.Conv1D(filters, kernel_size=1, padding='same', activation='relu')
        
        # Additional deep layers to further process the combined output.
        self.deep_layers = []
        for _ in range(num_layers):
            self.deep_layers.append(
                tf.keras.layers.Conv1D(filters, kernel_size=3, padding='same', activation='relu',
                                         kernel_regularizer=regularizers.l2(self.lambda_))
            )
        
        # Final projection to match the required output dimensions (if needed)
        self.final_projection = tf.keras.layers.Dense(units=filters)  # adjust if necessary

    @tf.function
    def call(self, input_tensor, training=False):
        def diag_zero(x):
            return tf.linalg.set_diag(x, self.diagonal)
        
        multi_scale_outputs = []
        for k_size in self.kernel_sizes:
            layers_for_scale = self.all_layers[k_size]
            x_scale = layers_for_scale[0](input_tensor)
            if self.q > 1:
                for i in range(1, self.q):
                    x_scale += layers_for_scale[i](tf.math.pow(input_tensor, i + 1))
            multi_scale_outputs.append(x_scale)
        
        x = tf.concat(multi_scale_outputs, axis=-1)
        x = self.combine_layer(x)
        
        # Pass through additional deep layers
        for layer in self.deep_layers:
            x = layer(x)
        
        if self.activation is not None:
            x = eval('tf.nn.' + self.activation + '(x)')
        
        x = tf.vectorized_map(fn=diag_zero, elems=x)
        x = tf.keras.layers.ActivityRegularization(l1=0.01)(x) 
        # Final projection if needed
        x = self.final_projection(x)
        
        return x
