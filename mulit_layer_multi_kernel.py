import tensorflow as tf
from tensorflow.keras import layers, regularizers

class MultiKernelEncoder(tf.keras.Model):
    def __init__(self, n, num_conv_layers=2, activation='tanh', merge_strategy="conv"):
        """
        :param n: Size of the input/output matrix (N x N)
        :param num_conv_layers: Number of Conv2D layers in each encoder
        :param activation: Activation function
        :param merge_strategy: "multiply" for element-wise multiplication, "conv" for Conv2D fusion
        """
        super(MultiKernelEncoder, self).__init__(name='MultiKernelEncoder')
        self.n = n
        self.num_conv_layers = num_conv_layers
        self.activation = activation
        self.merge_strategy = merge_strategy

        # Define three separate multi-layer encoders with different kernel sizes
        self.encoder_3 = self.build_encoder(kernel_size=3)
        self.encoder_5 = self.build_encoder(kernel_size=5)
        self.encoder_7 = self.build_encoder(kernel_size=7)

        # Final merging layer if using "conv" strategy
        if merge_strategy == "conv":
            self.merge_layer = layers.Conv2D(filters=1, kernel_size=3, padding="same", activation=activation)

    def build_encoder(self, kernel_size):
        """Creates a multi-layer encoder with the specified kernel size"""
        layers_list = []
        for _ in range(self.num_conv_layers):
            layers_list.append(layers.Conv2D(filters=16, kernel_size=kernel_size, padding="same", activation=self.activation))
        return tf.keras.Sequential(layers_list)

    def call(self, inputs):
        """
        Forward pass: Applies three encoders with different kernel sizes and combines the outputs
        """
        # Apply encoders separately
        op1 = self.encoder_3(inputs)
        op2 = self.encoder_5(inputs)
        op3 = self.encoder_7(inputs)

        if self.merge_strategy == "multiply":
            output = op1 * op2 * op3  # Element-wise multiplication
        elif self.merge_strategy == "conv":
            merged = tf.concat([op1, op2, op3], axis=-1)  # Concatenate along channel dimension
            output = self.merge_layer(merged)  # Apply final 2D convolution
        else:
            raise ValueError("Invalid merge_strategy. Use 'multiply' or 'conv'.")

        return output

# Example Usage
 )  # Should be (8, 64, 64, 1)
