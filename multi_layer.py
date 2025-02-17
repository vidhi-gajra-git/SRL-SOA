import tensorflow as tf

class Attention1D(tf.keras.layers.Layer):
    def __init__(self, filters):
        super(Attention1D, self).__init__()
        self.filters = filters
        self.query = tf.keras.layers.Conv1D(filters, kernel_size=1, activation=None)
        self.key = tf.keras.layers.Conv1D(filters, kernel_size=1, activation=None)
        self.value = tf.keras.layers.Conv1D(filters, kernel_size=1, activation=None)
        self.softmax = tf.keras.layers.Softmax(axis=1)

    def call(self, inputs):
        Q = self.query(inputs)  # Shape: [batch, length, filters]
        K = self.key(inputs)    # Shape: [batch, length, filters]
        V = self.value(inputs)  # Shape: [batch, length, filters]

        attention_weights = self.softmax(tf.matmul(Q, K, transpose_b=True) / tf.sqrt(float(self.filters)))
        output = tf.matmul(attention_weights, V)  # Shape: [batch, length, filters]

        return output + inputs  # Residual connection

class Oper1DMultiScaleCombined2(tf.keras.Model):
    def __init__(self, filters, kernel_sizes, activation='relu', q=1):
        super(Oper1DMultiScaleCombined, self).__init__()
        self.filters = filters
        self.kernel_sizes = kernel_sizes
        self.q = q
        self.activation = activation
        
        # 1D Attention Layer (Added before self-representation layers)
        self.attention = Attention1D(filters)
        
        # Multi-scale convolutional layers
        self.all_layers = {}
        for k_size in kernel_sizes:
            layers_for_scale = []
            for i in range(q):
                layers_for_scale.append(
                    tf.keras.layers.Conv1D(filters, k_size, padding='same', activation='relu')
                )
            self.all_layers[k_size] = layers_for_scale
        
        # Self-representation layer
        self.combine_layer = tf.keras.layers.Conv1D(filters, kernel_size=1, padding='same', activation='relu')

    def call(self, input_tensor):
        # Apply 1D Attention first
        attention_out = self.attention(input_tensor)

        multi_scale_outputs = []
        for k_size in self.kernel_sizes:
            layers_for_scale = self.all_layers[k_size]
            x_scale = layers_for_scale[0](attention_out)
            if self.q > 1:
                for i in range(1, self.q):
                    x_scale += layers_for_scale[i](tf.math.pow(attention_out, i + 1))
            multi_scale_outputs.append(x_scale)
        
        x = tf.concat(multi_scale_outputs, axis=-1)
        x = self.combine_layer(x)

        return x
