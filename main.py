import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ['TF_DETERMINISTIC_OPS'] = '1'

import numpy as np
import tensorflow as tf
import argparse

import svm
import rf
import utils

np.random.seed(10)
tf.random.set_seed(10)

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

y_predict = []
n=6
print("*"*10," METHOD : SVM","*"*10)
# Band selection ...
print("\t"*5,"*"*5,f" #RUNS : {n} ","*"*5)
for i in range(0, 3): # 10 runs ...
    

    if param['modelType'] != 'None':
        classData[i], Data[i] = utils.reduce_bands(param, classData[i], Data[i], i)    

    print('Classification...')
    if parameterSearch:
        # If hyper-parameter search is selected.
        best_parameters, class_model = svm.svm_train_search(classData[i]['x_train'], classData[i]['y_train'])
        print('\nBest paramters:' + str(best_parameters))
    else:
        class_model = svm.svm_train(classData[i]['x_train'], classData[i]['y_train'])
    
    

    y_predict.append(class_model.predict(classData[i]['x_test']))
    utils.evalPerformance(classData, y_predict,i+1)
for i in range(3, 7): # 10 runs ...
    
    if i<6 and param['modelType'] != 'None':
        classData[i], Data[i] = utils.reduce_bands(param, classData[i], Data[i], i)    

    print('Classification...')
    if parameterSearch:
        # If hyper-parameter search is selected.
        if i==6:
           print("$"*10,"Note the final o/p is tested on all bands","$"*10)
        best_parameters, class_model = svm.svm_train_search(classData[i]['x_train'], classData[i]['y_train'])
        print('\nBest paramters:' + str(best_parameters))
    else:
        class_model = svm.svm_train(classData[i]['x_train'], classData[i]['y_train'])
    

    y_predict.append(class_model.predict(classData[i]['x_test']))
    utils.evalPerformance(classData, y_predict,i+1)
    


    

# utils.evalPerformance(classData, y_predict)
# classData, Data = utils.loadData(param['dataset'])


# Comparing with random forest

