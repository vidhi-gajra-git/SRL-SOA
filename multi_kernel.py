import tensorflow as tf

class Oper1DMultiScaleCombined(tf.keras.Model):
    def __init__(self, filters, kernel_sizes, activation='relu', q=1):
        """
        Parameters:
          - filters: Number of filters for each Conv1D per scale.
          - kernel_sizes: List of kernel sizes, e.g., [3, 5, 7].
          - activation: Activation function to apply (e.g., 'relu').
          - q: Number of convolutional layers (order) per scale.
        """
        super(Oper1DMultiScaleCombined, self).__init__(name='')
        self.diagonal = tf.zeros(filters)
        self.activation = activation
        self.q = q
        self.kernel_sizes = kernel_sizes

        # Create multi-scale Conv1D layers for each kernel size.
        self.all_layers = {}
        for k_size in kernel_sizes:
            layers_for_scale = []
            for i in range(q):
                layers_for_scale.append(
                    tf.keras.layers.Conv1D(filters, k_size, padding='same', activation=None, kernel_regularizer=tf.keras.regularizers.l2(0.01))
                )
            self.all_layers[k_size] = layers_for_scale
        
        # Layer Normalization instead of BatchNormalization
        self.layer_norm = tf.keras.layers.LayerNormalization(axis=-1)  # Normalize across features axis.
        
        # Dropout (if needed)
        # self.dropout = tf.keras.layers.Dropout(0.5)

        # 1x1 convolution layer to combine multi-scale features back to 'filters' channels.
        self.combine_layer = tf.keras.layers.Conv1D(filters, kernel_size=1, padding='same', activation=None)
    
    @tf.function
    def call(self, input_tensor, training=False):
        def diag_zero(x):
            return tf.linalg.set_diag(x, self.diagonal)
        
        multi_scale_outputs = []
        # Process input with each scale.
        for k_size in self.kernel_sizes:
            layers_for_scale = self.all_layers[k_size]
            x_scale = self.layer_norm(layers_for_scale[0](input_tensor))
            if self.q > 1:
                for i in range(1, self.q):
                    x_scale += self.layer_norm(layers_for_scale[i](tf.math.pow(input_tensor, i + 1)))
            # x_scale = self.dropout(x_scale)
            multi_scale_outputs.append(x_scale)
        
        # Concatenate outputs along the channel dimension.
        x = tf.concat(multi_scale_outputs, axis=-1)  # Shape: [B, L, filters * num_scales]
        
        # Use a 1x1 convolution to reduce the number of channels.
        x = self.combine_layer(x)  # Now shape becomes: [B, L, filters]
        
        # Optionally apply activation.
        if self.activation is not None:
            x = eval('tf.nn.' + self.activation + '(x)')
        
        # Apply diagonal constraint (if needed).
        x = tf.vectorized_map(fn=diag_zero, elems=x)
        
        return x
