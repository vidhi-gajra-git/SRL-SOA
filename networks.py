import tensorflow as tf
import numpy as np

from layers import Oper1D
from multi_kernel import Oper1DMultiScaleCombined
np.random.seed(10)
tf.random.set_seed(10)

### SLR-OL
def SLRol(n_bands, q):
  input = tf.keras.Input((n_bands, 1), name='input')
  # x_0 = Oper1D(n_bands, 3, activation = 'tanh', q = q)(input)
  x_0 =Oper1DMultiScaleCombined(n_bands, [3,5,9], activation = 'tanh', q = q)(input)
  
  # testing the model on multi-scale conv .... and comparing the o/p 
  y = tf.keras.layers.Dot(axes=(2,1))([x_0, input])

  model = tf.keras.models.Model(input, y, name='OSEN')

  # optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
  # Adjust the learning rate
  optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
  model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])

# Add early stopping
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

  history = model.fit(train_data, train_labels, validation_data=(val_data, val_labels), epochs=100, callbacks=[early_stopping])



  # model.compile(optimizer = optimizer, loss = 'mse')

  model.summary()
    
  return model
