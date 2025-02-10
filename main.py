import os
import numpy as np
import tensorflow as tf
import argparse

import svm
import utils

# Set the GPU device explicitly (optional)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Choose GPU 0, change to another number for different GPUs

np.random.seed(10)
tf.random.set_seed(10)

# Enable memory growth for GPUs if running out of memory
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently only one GPU available, make sure TensorFlow uses it efficiently
        tf.config.set_logical_device_configuration(gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=4096)])  # Limit memory to 4GB
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print("Error setting memory growth: ", e)
else:
    print("No GPU found. The code will run on CPU.")

# Argument parsing
ap = argparse.ArgumentParser()
ap.add_argument('--method', default='SRL-SOA', help="SRL-SOA, PCA, SpaBS, EGCSR_R, ISSC, None (for no band selection).")
ap.add_argument('--dataset', default='Indian_pines_corrected', help="Indian_pines_corrected, SalinasA_corrected.")
ap.add_argument('--q', default=3, help="Order of the OSEN.")
ap.add_argument('--weights', default=True, help="Evaluate the model.")
ap.add_argument('--epochs', default=50, help="Number of epochs.")
ap.add_argument('--batchSize', default=5, help="Batch size.")
ap.add_argument('--bands', default=1, help="Compression rate.")
args = vars(ap.parse_args())

param = {}

param['modelType'] = args['method']
param['weights'] = args['weights']  # True or False.
param['q'] = int(args['q'])  # The order of the OSEN.
param['dataset'] = args['dataset']  # Dataset.
param['epochs'] = int(args['epochs'])
param['batchSize'] = int(args['batchSize'])
param['s_bands'] = int(args['bands'])  # Number of bands.
parameterSearch = True  # Parameter search for the classifier.

# Load data
classData, Data = utils.loadData(param['dataset'])

y_predict = []

# Band selection ...
for i in range(0, 3):  # 10 runs ...
    if param['modelType'] != 'None':
        classData[i], Data[i] = utils.reduce_bands(param, classData[i], Data[i], i)    

    print('Classification...')
    if parameterSearch:
        # If hyper-parameter search is selected.
        best_parameters, class_model = svm.svm_train_search(classData[i]['x_train'], classData[i]['y_train'])
        print('\nBest parameters:' + str(best_parameters))
    else:
        class_model = svm.svm_train(classData[i]['x_train'], classData[i]['y_train'])

    y_predict.append(class_model.predict(classData[i]['x_test']))

# Evaluate performance
utils.evalPerformance(classData, y_predict)
