import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers

class SelfONN1D(Model):
    def __init__(self, filters, kernel_size, q=2, num_self_neurons=1, dilation_rate=1, lambda_l1=0.001, lambda_l2=0.001):
        super(SelfONN1D, self).__init__()

        self.q = q  # Order of non-linearity

        # First Conv1D Layer
        self.conv1 = layers.Conv1D(
            filters, kernel_size, padding='same', 
            activation=None, dilation_rate=dilation_rate,
            kernel_regularizer=regularizers.l1_l2(l1=lambda_l1, l2=lambda_l2)
        )

        # Second Conv1D Layer (Non-Linear Neurons)
        self.conv2 = layers.Conv1D(
            filters, kernel_size, padding='same', 
            activation=None, dilation_rate=dilation_rate,
            kernel_regularizer=regularizers.l1_l2(l1=lambda_l1, l2=lambda_l2)
        )

        # Self-Representation Neuron
        # self.self_representation = layers.Dense(
        #     num_self_neurons, activation='sigmoid', name="Self_Representation",
        #     kernel_regularizer=regularizers.l1_l2(l1=lambda_l1, l2=lambda_l2)
        # )

        # Final Combining Layer
        # self.output_layer = layers.Conv1D(
        #     filters, 1, padding='same', activation='linear',
        #     kernel_regularizer=regularizers.l1_l2(l1=lambda_l1, l2=lambda_l2)
        # )

        # Activation Functions
        self.activation = tf.keras.layers.LeakyReLU(alpha=0.1)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.activation(x)

        # Apply non-linear transformations with q-order terms
        x_q = x
        for i in range(2, self.q + 1):
            x_q += tf.math.pow(x, i)

        x_q = self.conv2(x_q)
        x_q = self.activation(x_q)

        # Self-Representation Neuron
        # self_repr = self.self_representation(x_q)

        # Adaptive weighting
        # x = x_q * self_repr

        # Final Output Layer
        # x = self.output_layer(x)
        print(f"Final shape of x is {x.shape}")
        return x_q

# # Model Instantiation
# filters = 32
# kernel_size = 3
# q_order = 3  # Non-linear order
# model = SelfONN1D(filters=filters, kernel_size=kernel_size, q=q_order)

# # Build Model (Example Input Shape: (Batch, Time Steps, Channels))
# model.build(input_shape=(None, 100, 1))
# model.summary()
