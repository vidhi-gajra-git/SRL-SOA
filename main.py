import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ['TF_DETERMINISTIC_OPS'] = '1'
# setting up gpu 

import numpy as np
import tensorflow as tf
import argparse

import svm
import rf
import utils
# Restrict TensorFlow to only use GPU 0
import tensorflow as tf
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("$"*10,"GPUs Available:", tf.config.list_physical_devices('GPU'),"$"*10)
    try:
        for gpu in gpus:
            
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU is configured for TensorFlow!")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU found, running on CPU.")



# tf.config.set_visible_devices(tf.config.list_physical_devices('GPU')[0], 'GPU')

# Allow TensorFlow to dynamically allocate GPU memory
tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
np.random.seed(42)
tf.random.set_seed(42)
np.random.seed(42)
# np.random.set_seed(42)
ap = argparse.ArgumentParser()
ap.add_argument('--method', default='SRL-SOA', help =
                "SRL-SOA, PCA, SpaBS, EGCSR_R, ISSC, None (for no band selection).")
ap.add_argument('--dataset', default='Indian_pines_corrected', help = "Indian_pines_corrected, SalinasA_corrected.")
ap.add_argument('--q', default = 3, help = "Order of the OSEN.")
ap.add_argument('--weights', default = True, help="Evaluate the model.")
ap.add_argument('--epochs', default = 50, help="Number of epochs.")
ap.add_argument('--batchSize', default = 5, help="Batch size.")
ap.add_argument('--bands', default = 1, help="Compression rate.")
args = vars(ap.parse_args())

param = {}

param['modelType'] = args['method']
param['weights'] = args['weights'] # True or False.
param['q'] = int(args['q']) # The order of the OSEN.
param['dataset'] = args['dataset'] # Dataset.
param['epochs'] = int(args['epochs'])
param['batchSize'] = int(args['batchSize'])
param['s_bands'] = int(args['bands']) # Number of bands.
parameterSearch = True # Parameter search for the classifier.

classData, Data = utils.loadData(param['dataset'])
classDataEval , DataEval= utils.loadEvalData(param['dataset'])
y_predict = []
n=6
print("*"*10," METHOD : SVM","*"*10)
# Band selection ...
print("\t"*5,"*"*5,f" #RUNS : {n} ","*"*5)
for i in range(0, 3): # 10 runs ...
    

    if param['modelType'] != 'None':
        selected_bands=utils.reduce_bands(param, classData[i], Data[i], i)  
        # classDataEval[i]=classDataEval[i]
        # Data[i] = utils.reduce_bands(param, classData[i], Data[i], i)    

    print('Classification...')
    if parameterSearch:
        # If hyper-parameter search is selected.
        best_parameters, class_model = svm.svm_train_search(classDataEval[i]['x_train'][:,selected_bands ], classDataEval[i]['y_train'][:,selected_bands ])
        print('\nBest paramters:' + str(best_parameters))
    else:
        class_model = svm.svm_train(classDataEval[i]['x_train'][:,selected_bands ], classDataEval[i]['y_train'][:,selected_bands ])
    
    

    y_predict.append(class_model.predict(classDataEval[i]['x_test'][:,selected_bands ]))
    utils.evalPerformance(classDataEval, y_predict,i+1)
for i in range(3, 7): # 10 runs ...
    
    if i<6 and param['modelType'] != 'None':
                selected_bands=utils.reduce_bands(param, classData[i], Data[i], i)     

    print('Classification...')
    if parameterSearch:
        # If hyper-parameter search is selected.
        if i==6:
           print("$"*10,"Note the final o/p is tested on all bands","$"*10)
        best_parameters, class_model = svm.svm_train_search(classDataEval[i]['x_train'][:,selected_bands ], classDataEval[i]['y_train'][:,selected_bands ])
        print('\nBest paramters:' + str(best_parameters))
    else:
         class_model = svm.svm_train(classDataEval[i]['x_train'][:,selected_bands ], classDataEval[i]['y_train'][:,selected_bands ])
    


    y_predict.append(class_model.predict(classDataEval[i]['x_test'][:,selected_bands ]))
    utils.evalPerformance(classDataEval, y_predict,i+1)
    


    

# utils.evalPerformance(classData, y_predict)
# classData, Data = utils.loadData(param['dataset'])


# Comparing with random forest

