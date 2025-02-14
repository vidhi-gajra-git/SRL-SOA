import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers

class Oper1DMultiScaleCombined(tf.keras.Model):
    def __init__(self, filters, kernel_sizes, activation='relu', q=1):
        super(Oper1DMultiScaleCombined, self).__init__(name='')
        self.diagonal = tf.zeros(filters)
        self.activation = activation
        self.q = q
        self.kernel_sizes = kernel_sizes
        self.lambda_=0.001
        self.all_layers = {}
       

        for k_size in kernel_sizes:
            layers_for_scale = []
            for i in range(q):
                layers_for_scale.append(
                    tf.keras.layers.Conv1D(filters, k_size, padding='same', activation='relu', )
                )
            
                # kernel_regularizer=regularizers.l2(lambda_/2),activity_regularizer=sparse_regularizer
            self.all_layers[k_size] = layers_for_scale
        
        # Dropout and BatchNormalization removed (or replaced if needed)
        # self.dropout = tf.keras.layers.Dropout(0.2)
        self.combine_layer = tf.keras.layers.Conv1D(filters, kernel_size=1, padding='same', activation='relu')
   
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
            # x_scale = self.dropout(x_scale)
            multi_scale_outputs.append(x_scale)
        
        x = tf.concat(multi_scale_outputs, axis=-1)
        x = self.combine_layer(x)
        
        if self.activation is not None:
            x = eval('tf.nn.' + self.activation + '(x)')
        
        x = tf.vectorized_map(fn=diag_zero, elems=x)
        
        return x

