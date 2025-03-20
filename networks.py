

import tensorflow as tf
import numpy as np

from layers import Oper1D
from multi_kernel import Oper1DDilated
from multi_layer import SparseAutoencoderWithAttention
from self_onn import SparseAutoencoderNonLinear
from tensorflow.keras import backend as K

# from multi_layer_multi_kernel import SparseAutoencoderNonLinear2 ,MultiKernelEncoder 
np.random.seed(42)
tf.random.set_seed(42)


### SLR-OL
def SLRol(n_bands, q):
  input = tf.keras.Input((n_bands, 1), name='input')


# Load the pretrained guiding classifier
  guiding_classifier = tf.keras.models.load_model("guiding_classifier_KSC.h5")

# Ensure the classifier is not trainable
  guiding_classifier.trainable = False  

# Print model summary to verify
  guiding_classifier.summary()

  # x_0 = Oper1D(n_bands, 3, activation = 'tanh', q = q)(input)
  # model_name=f'Oper1D_q{q}_KSC'
  # hyperparams = Oper1D(n_bands, 3, activation = 'tanh', q = q).get_hyperparameters()
 
  # q = 3    # Degree of non-linearity
  num_conv_layers = 2 # Number of Conv1D layers per degree
  x_0= SparseAutoencoderNonLinear(n=n_bands, q=q, num_conv_layers=num_conv_layers, activation='tanh')(input)
  model_name=f'SparseAutoencoderNonLinear{q}_layers{num_conv_layers}_Xavier_'
  hyperparams = SparseAutoencoderNonLinear(n=n_bands, q=q, num_conv_layers=num_conv_layers,activation='tanh').get_hyperparameters()
  # x_0=MultiKernelEncoder(n=n_bands, q=q, num_conv_layers=num_conv_layers)(input)
  # model_name=f'MultiKernelEncoder{q}_layers{num_conv_layers}_Xavier_init_3_5_7'
  # hyperparams = MultiKernelEncoder(n=n_bands, q=q, num_conv_layers=num_conv_layers).get_hyperparameters()

 def sparse_loss(y_true, y_pred):
        sparsity_level = 0.01
        mse_loss = tf.math.reduce_mean(tf.keras.losses.MSE(y_true, y_pred))
        mean_activation = tf.math.reduce_mean(x_0, axis=0)  # Use the existing output

        kl_divergence = tf.reduce_sum(sparsity_level * tf.math.log(sparsity_level / (mean_activation + 1e-10)) +
                                      (1 - sparsity_level) * tf.math.log((1 - sparsity_level) / (1 - mean_activation + 1e-10)))

        return mse_loss + lambda_sparse * kl_divergence

  # # print("!!!!!!!!!!",x_0.shape, "!!!!!!!!!!!!!!!!!!!")
  # x_0=SelfONN1D(filters=n_bands, kernel_size=5,q=q)
  # x_0 =Oper1DDilated(n_bands, dilation_rates=[1, 2, 4], activation = 'tanh', q = q)(input)
  # x_0 = SparseAutoencoderWithAttention(n_bands, [3,5,9], activation = 'tanh', q = q)(input)
  
  # testing the model on multi-scale conv .... and comparing the o/p 
  y = tf.keras.layers.Dot(axes=(2,1))([x_0, input])
  class_preds = guiding_classifier(y)

# # # Define the combined model
  combined_model = tf.keras.Model(inputs=input, outputs=[y, class_preds], name='GuidedSparseAutoencoder')
  model_name=f'CombinedModel{q}_Xavier_Classifier_KSC_run2'
   

# Compile the model

# Print summary
  # combined_model.summary()

  # model = tf.keras.models.Model(input, y, name=model_name)
  
#   lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#     initial_learning_rate=0.001,
#     decay_steps=10000,
#     decay_rate=0.9
# )

  # lr_schedule = WarmUpExponentialDecay(initial_lr=hyperparams2["initial_lr"],
  #                                        warmup_steps=hyperparams2["warmup_steps"],
  #                                        decay_steps=hyperparams2["decay_steps"],
  #                                        decay_rate=hyperparams2["decay_rate"])

  # optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule )
  # optimizer = tf.keras.optimizers.RMSProp(learning_rate=0.001)
  # optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001)
  optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

#   # Adjust the learning rate
#   optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
  # model.compile(optimizer=optimizer,loss='mse')
  combined_model.compile(optimizer=optimizer, loss=[sparse_loss, 'categorical_crossentropy'], loss_weights=[1.0, 1.0])
  

# # Add early stopping
  



  # model.compile(optimizer = optimizer, loss = 'mse')
  

  # combined_model.summary()
    
  return model_name, hyperparams , combined_model
