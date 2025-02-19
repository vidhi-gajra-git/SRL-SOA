
import tensorflow as tf
import numpy as np

from layers import Oper1D
from multi_kernel import Oper1DDilated
from multi_layer import SparseAutoencoderWithAttention
from self_onn import SelfONN1D
np.random.seed(10)
tf.random.set_seed(10)

### SLR-OL
def SLRol(n_bands, q):
  input = tf.keras.Input((n_bands, 1), name='input')
  # x_0 = Oper1D(n_bands, 3, activation = 'tanh', q = q)(input)
  x_0=SelfONN1D(filters=n_bands, kernel_size=5,q=q)
  # x_0 =Oper1DDilated(n_bands, dilation_rates=[1, 2, 4], activation = 'tanh', q = q)(input)
  # x_0 = SparseAutoencoderWithAttention(n_bands, [3,5,9], activation = 'tanh', q = q)(input)
  
  # testing the model on multi-scale conv .... and comparing the o/p 
  y = tf.keras.layers.Dot(axes=(2,1))([x_0, input])

  model = tf.keras.models.Model(input, y, name='OSEN')

  # optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
  # Adjust the learning rate
  optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
  model.compile(optimizer=optimizer,loss=tf.keras.losses.Huber(delta=1.0))

# Add early stopping
  



  # model.compile(optimizer = optimizer, loss = 'mse')

  model.summary()
    
  return model
