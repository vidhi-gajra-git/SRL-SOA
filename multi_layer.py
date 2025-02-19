import tensorflow as tf
from tensorflow.keras import regularizers

class SparseAutoencoderWithAttention(tf.keras.Model):
    def __init__(self, filters, kernel_sizes, activation='relu', q=1):
        super(SparseAutoencoderWithAttention, self).__init__(name='SparseAutoencoder')
        self.diagonal = tf.zeros(filters)
        self.activation = activation
        self.q = q
        self.kernel_sizes = kernel_sizes
        self.lambda_ = 0.001
        self.all_layers = {}
        self.alpha=0.01
        # Multi-scale Conv1D layers
        for k_size in kernel_sizes:
            layers_for_scale = []
            for i in range(q):
                layers_for_scale.append(
                    tf.keras.layers.Conv1D(filters, k_size, padding='same', activation=None)
                )
            self.all_layers[k_size] = layers_for_scale
        
        # Self-Attention Layer
        self.attention_layer = tf.keras.layers.Attention()

        # Final combining layer
        self.combine_layer = tf.keras.Sequential([
            tf.keras.layers.Conv1D(filters, kernel_size=1, padding='same', activation=None),
            tf.keras.layers.LeakyReLU(alpha=self.alpha)  # LeakyReLU for the final layer
        ])

        # Activity Regularization for Sparsity
        # self.sparse_reg = tf.keras.layers.ActivityRegularization(l1=0.01)
   
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
        
        # Concatenating multi-scale outputs
        x = tf.concat(multi_scale_outputs, axis=-1)

        # Apply Self-Attention
        x = self.attention_layer([x, x])  # Query = Key = Value

        # Sparsity Regularization
        x = tf.keras.layers.ActivityRegularization(l1=0.001)(x)

        # Combine and apply activation
        x = self.combine_layer(x)
        
        if self.activation is not None:
            x = eval('tf.nn.' + self.activation + '(x)')
        
        # Diagonal zeroing
        x = tf.vectorized_map(fn=diag_zero, elems=x)

        return x
