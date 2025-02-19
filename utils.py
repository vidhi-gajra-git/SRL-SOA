import os
import scipy.io
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, cohen_kappa_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, cohen_kappa_score
from tabulate import tabulate  # Install with: pip install tabulate
import networks
import ipykernel
import pandas as pd
from IPython.display import display
try:
    from BandSelection.classes.SpaBS import SpaBS
    from BandSelection.classes.ISSC import ISSC_HSI
    from GCSR_BS.EGCSR_BS_Ranking import EGCSR_BS_Ranking as EGCSR_R
except ModuleNotFoundError:
    pass

np.random.seed(10)
tf.random.set_seed(10)

def loadData(dataset):
    Data = scipy.io.loadmat('data/' + dataset + '.mat')
    if 'Indian' in dataset: Gtd = scipy.io.loadmat('data/' + 'Indian_pines_gt.mat')
    elif 'SalinasA' in dataset: Gtd = scipy.io.loadmat('data/' + 'SalinasA_gt.mat')
    else: Gtd = scipy.io.loadmat('data/' + dataset + '_gt.mat')

    if dataset == 'Indian_pines_corrected':
        image = Data['indian_pines_corrected']
        gtd = Gtd['indian_pines_gt']
    elif dataset == 'SalinasA_corrected':
        image = Data['salinasA_corrected']
        gtd = Gtd['salinasA_gt']
    else: print('The selected dataset is not valid.')

    image = np.array(image, dtype = 'float32')
    gtd = np.array(gtd, dtype = 'float32')

    xx = np.reshape(image, [image.shape[0] * image.shape[1], image.shape[2]])

    # Classification data.
    label = np.reshape(gtd, [gtd.shape[0] * gtd.shape[1]])

    x_class = xx[label != 0]
    y_class = label[label != 0]

    classDataa = []
    Dataa = []
    for i in range(0, 10):

        classData = {}
        Data = {}

        if dataset == 'Indian_pines_corrected':
            x_train, x_test, y_train, y_test = train_test_split(x_class, y_class,
            test_size = 0.8, random_state = i + 1)
        else:
            x_train, x_test, y_train, y_test = train_test_split(x_class, y_class,
            test_size = 0.85, random_state = i + 1)

        scaler = StandardScaler().fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)

        classData['x_train'] = x_train
        classData['x_test'] = x_test
        classData['y_train'] = y_train - 1
        classData['y_test'] = y_test - 1

        del x_train, x_test, y_train, y_test

        # Image data.

        sc = np.reshape(image, [image.shape[0] * image.shape[1], image.shape[2]])

        sc = scaler.transform(sc)

        scd = sc[label == 0]

        sc = np.reshape(sc, [image.shape[0], image.shape[1], image.shape[2]])

        Data['scd'] = scd
        Data['sc'] = sc
        Data['gtd'] = gtd

        classDataa.append(classData)
        Dataa.append(Data)

    print('\nScene: ', sc.shape)

    
    print('\nClassification:')
    print('Training samples: ', len(classData['x_train']))
    print('Test samples: ', len(classData['x_test']))

    print('\n')
    print('Number of bands: ', str(classData['x_train'].shape[-1]))
    

    return classDataa, Dataa

def reduce_bands(param, classData, Data, i):
    modelType = param['modelType']
    dataset = param['dataset']
    q = param['q']
    weights = param['weights']
    batchSize = param['batchSize']
    epochs = param['epochs']
    s_bands = param['s_bands']
    q = param['q']
    ind_a=None

    n_bands = classData['x_train'].shape[-1]

    if dataset != 'SalinasA_corrected': xx = classData['x_train']
    else: xx = np.concatenate([classData['x_train'], Data['scd']], axis = 0)

    if modelType == 'SRL-SOA':
        weightsDir = 'weights/' + dataset + '/'
        if not os.path.exists(weightsDir): os.makedirs(weightsDir)
        weightName = weightsDir + modelType + '_q' + str(q) + '_run' + str(i) + '.weights.h5'
        model = networks.SLRol(n_bands = n_bands, q = q)

        checkpoint_osen = tf.keras.callbacks.ModelCheckpoint(
            weightName, monitor='val_loss', verbose=1,
            save_best_only=True, mode='min', save_weights_only=True)

        callbacks_osen = [checkpoint_osen]

        if weights == 'False':
            model.fit(xx, xx, batch_size = batchSize,
                    callbacks=callbacks_osen, shuffle=True,
                    validation_data=(xx, xx), epochs = epochs)
            print(modelType + ' is trained!')

        intermediate_layer_model = tf.keras.Model(inputs = model.input,
                                        outputs = model.layers[1].output)
        A = intermediate_layer_model(classData['x_train'])

        A = np.abs(A)
        A = np.mean(A, axis = 0)
        A = np.sum(A, axis = 0)
        indices = np.argsort(A)
        ind_a=indices[-s_bands::]
        df_bands = pd.DataFrame(ind_a, columns=["Selected_Bands"])
        csv_file_path = "results/selected_bands.csv"
        
        # Check if the file exists
        band_presence = []
        all_bands=[i for i in range (len(indices))]
        for band in all_bands:
            band_presence.append(1 if band in ind_a else 0)
    
    # Create a DataFrame for the selected bands and their presence
        df_bands = pd.DataFrame([band_presence], columns=all_bands)

        classData['x_train'] = classData['x_train'][:, indices[-s_bands::]]
        classData['x_test'] = classData['x_test'][:, indices[-s_bands::]]

    elif modelType == 'PCA':
        pca = PCA(n_components = s_bands, random_state = 1)
        pca.fit(xx)
        classData['x_train'] = pca.transform(classData['x_train'])
        classData['x_test'] = pca.transform(classData['x_test'])

    elif modelType == 'SpaBS':
        model = SpaBS(s_bands)
        x_temp = model.predict(xx)

        a = xx[0, :]
        b = x_temp[0, :]
        band_indices, ind_a, ind_b = np.intersect1d(a, b, return_indices=True)

        classData['x_train'] = classData['x_train'][:, ind_a]
        classData['x_test'] = classData['x_test'][:, ind_a]

    elif modelType == 'EGCSR_R':
        model = EGCSR_R(s_bands, regu_coef=1e4, n_neighbors=5, ro=0.8)
        x_temp = model.predict(xx)

        a = xx[0, :]
        b = x_temp[0, :]
        band_indices, ind_a, ind_b = np.intersect1d(a, b, return_indices=True)

        classData['x_train'] = classData['x_train'][:, ind_a]
        classData['x_test'] = classData['x_test'][:, ind_a]

    elif modelType == 'ISSC':
        model = ISSC_HSI(s_bands, coef_=1.e-4)

        x_temp = model.predict(xx)

        a = xx[0, :]
        b = x_temp[0, :]
        band_indices, ind_a, ind_b = np.intersect1d(a, b, return_indices=True)

        classData['x_train'] = classData['x_train'][:, ind_a]
        classData['x_test'] = classData['x_test'][:, ind_a]

    else: print('Selected method is not supported.')

    print('Selected number of bands: ', str(classData['x_train'].shape[-1]))
    print(f'======Selected band indices ======= \n {ind_a}')

    return classData, Data

def evalPerformance(classData, y_predict,n):
     # Number of runs
    print(f"The model shall evaluate for {n} times")

    # Initialize arrays for storing accuracy scores
    oa = np.zeros(n, dtype='float64')
    aa = np.zeros(n, dtype='float64')
    kappa = np.zeros(n, dtype='float64')

    results = []
    confusion_matrices = []  # Store confusion matrices for CSV

    for i in range(n):
        y_test = classData[i]['y_test']
        cm = confusion_matrix(y_test, y_predict[i])

        oa[i] = np.sum(y_test == y_predict[i]) / len(y_predict[i])
        aa[i] = balanced_accuracy_score(y_test, y_predict[i])
        kappa[i] = cohen_kappa_score(y_test, y_predict[i])

        results.append([i + 1, oa[i], aa[i], kappa[i]])

        # Convert Confusion Matrix to Pandas DataFrame
        labels = sorted(set(y_test))  # Get unique class labels
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)

        print(f"\nConfusion Matrix for Run {i+1}:")
        display(cm_df)  # Use display() for Jupyter Notebook

        confusion_matrices.append(cm_df)  # Store for CSV output

        # Plot Confusion Matrix as Heatmap
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title(f"Confusion Matrix - Run {i+1}")
        display(plt.gcf())  # Display figure explicitly
        plt.close()  # Close figure to prevent memory issues

    # Convert results to Pandas DataFrame
    df_results = pd.DataFrame(results, columns=["Run", "Overall Accuracy", "Average Accuracy", "Kappa Coefficient"])

    # Save results to the "results" folder
    results_folder = "results"
    os.makedirs(results_folder, exist_ok=True)

    # Save accuracy results
    csv_path = os.path.join(results_folder, f"performance_results{i}.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"\nPerformance results saved to: {csv_path}")

    # Save confusion matrices
    cm_csv_path = os.path.join(results_folder, "confusion_matrices.csv")
    if os.path.exists(cm_csv_path):
        with open(cm_csv_path, "a") as f:
            for i, cm_df in enumerate(confusion_matrices):
                f.write(f"Confusion Matrix - Run {i+1}\n")
                cm_df.to_csv(f)
                f.write("\n")  # Add a newline between matrices
    else :
        with open(cm_csv_path, "w") as f:
            for i, cm_df in enumerate(confusion_matrices):
                f.write(f"Confusion Matrix - Run {i+1}\n")
                cm_df.to_csv(f)
                f.write("\n")
        
    print(f"Confusion matrices saved to: {cm_csv_path}")

    # Print Results Table
    print("\nPerformance Metrics Summary:")
    display(df_results)  # Use display() to properly render the table

    # Compute average performance metrics
    avg_oa = np.mean(oa)
    avg_aa = np.mean(aa)
    avg_kappa = np.mean(kappa)
    print(f"\nAverage Performance Over {n} Runs:")
    print(f"Overall Accuracy: {avg_oa:.4f}")
    print(f"Average Accuracy: {avg_aa:.4f}")
    print(f"Kappa Coefficient: {avg_kappa:.4f}")

    # Visualization: Plot different accuracy metrics across runs
    sns.set_style("whitegrid")
    plt.figure(figsize=(8, 5))

    x_labels = [f"Run {i+1}" for i in range(n)]
    width = 0.2  # Bar width

    plt.bar(np.arange(n), oa, width=width, label="Overall Accuracy", color="blue")
    plt.bar(np.arange(n) + width, aa, width=width, label="Average Accuracy", color="green")
    plt.bar(np.arange(n) + 2 * width, kappa, width=width, label="Kappa Coefficient", color="red")

    plt.axhline(avg_oa, color="blue", linestyle="dashed", linewidth=1, label="Avg OA")
    plt.axhline(avg_aa, color="green", linestyle="dashed", linewidth=1, label="Avg AA")
    plt.axhline(avg_kappa, color="red", linestyle="dashed", linewidth=1, label="Avg Kappa")

    plt.xticks(np.arange(n) + width, x_labels)
    plt.ylabel("Score")
    plt.title("Performance Metrics Across Runs")
    plt.legend()
    
    display(plt.gcf())  # Explicitly display the figure
    plt.close()  # Close the figure to prevent file access issues
