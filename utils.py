

import os
import scipy.io
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, cohen_kappa_score , classification_report
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
# from sklearn.metrics import confusion_matrix, balanced_accuracy_score, cohen_kappa_score
from tabulate import tabulate  # Install with: pip install tabulate
import networks
import ipykernel
import pandas as pd
from IPython.display import display
import mlflow
import dagshub
import matplotlib 
import time
import psutil
# import platform

# %matplotlib inline
try:
    from BandSelection.classes.SpaBS import SpaBS
    from BandSelection.classes.ISSC import ISSC_HSI
    from GCSR_BS.EGCSR_BS_Ranking import EGCSR_BS_Ranking as EGCSR_R
except ModuleNotFoundError:
    pass
matplotlib.use('Agg')
np.random.seed(42)
tf.random.set_seed(42)
# Connect MLflow to DagsHub
dagshub.init(repo_owner='vidhi-gajra-git', repo_name='SRL_SOA', mlflow=True)
# dagshub.auth.add_credentials(
#     username="your-dagshub-username",
#     token="your-dagshub-access-token"
# )

mlflow.set_tracking_uri("https://dagshub.com/vidhi-gajra-git/SRL_SOA.mlflow")
mlflow.set_experiment("SRL_SOA_V.1")
# Capture System Metrics
# cpu_info = platform.processor()
cpu_count = psutil.cpu_count(logical=True)
ram_total = round(psutil.virtual_memory().total / (1024 ** 3), 2)  # GB

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import scipy.io
def loadEvalData(dataset):
    np.random.seed(42)  # Set global seed

    Data = scipy.io.loadmat('data/' + dataset + '.mat')
    if 'Indian' in dataset:
        Gtd = scipy.io.loadmat('data/' + 'Indian_pines_gt.mat')
    elif 'SalinasA' in dataset:
        Gtd = scipy.io.loadmat('data/' + 'SalinasA_gt.mat')
    else:
        Gtd = scipy.io.loadmat('data/' + dataset + '_gt.mat')

    if dataset == 'Indian_pines_corrected':
        image = Data['indian_pines_corrected']
        gtd = Gtd['indian_pines_gt']
    elif dataset == 'SalinasA_corrected':
        image = Data['salinasA_corrected']
        gtd = Gtd['salinasA_gt']
    else:
        raise ValueError('The selected dataset is not valid.')

    image = np.array(image, dtype='float32')
    gtd = np.array(gtd, dtype='float32')

    xx = np.reshape(image, [image.shape[0] * image.shape[1], image.shape[2]])
    label = np.reshape(gtd, [gtd.shape[0] * gtd.shape[1]])

    x_class = xx[label != 0]
    y_class = label[label != 0]

    # Fit a single scaler for consistency
    global_scaler = StandardScaler()
    global_scaler.fit(x_class)

    classDataa = []
    Dataa = []
    for _ in range(10):  # No need to vary seed in loop
        x_train, x_test, y_train, y_test = train_test_split(
            x_class, y_class, test_size=0.2, random_state=42
        )

        # Use the global scaler
        x_train = global_scaler.transform(x_train)
        x_test = global_scaler.transform(x_test)

        classData = {
            'x_train': x_train,
            'x_test': x_test,
            'y_train': y_train - 1,
            'y_test': y_test - 1
        }

        # Process image data with the same scaler
        sc = np.reshape(image, [image.shape[0] * image.shape[1], image.shape[2]])
        sc = global_scaler.transform(sc)
        scd = sc[label == 0]
        sc = np.reshape(sc, [image.shape[0], image.shape[1], image.shape[2]])

        Data = {'scd': scd, 'sc': sc, 'gtd': gtd}

        classDataa.append(classData)
        Dataa.append(Data)

    print('\nScene: ', sc.shape)
    print('\nClassification:')
    print('Training samples: ', len(classData['x_train']))
    print('Test samples: ', len(classData['x_test']))
    print('\nNumber of bands: ', str(classData['x_train'].shape[-1]))

    return classDataa, Dataa

def loadData(dataset):
    np.random.seed(42)  # Set global seed

    Data = scipy.io.loadmat('data/' + dataset + '.mat')
    if 'Indian' in dataset:
        Gtd = scipy.io.loadmat('data/' + 'Indian_pines_gt.mat')
    elif 'SalinasA' in dataset:
        Gtd = scipy.io.loadmat('data/' + 'SalinasA_gt.mat')
    else:
        Gtd = scipy.io.loadmat('data/' + dataset + '_gt.mat')

    if dataset == 'Indian_pines_corrected':
        image = Data['indian_pines_corrected']
        gtd = Gtd['indian_pines_gt']
    elif dataset == 'SalinasA_corrected':
        image = Data['salinasA_corrected']
        gtd = Gtd['salinasA_gt']
    else:
        raise ValueError('The selected dataset is not valid.')

    image = np.array(image, dtype='float32')
    gtd = np.array(gtd, dtype='float32')

    xx = np.reshape(image, [image.shape[0] * image.shape[1], image.shape[2]])
    label = np.reshape(gtd, [gtd.shape[0] * gtd.shape[1]])

    x_class = xx[label != 0]
    y_class = label[label != 0]

    # Fit a single scaler for consistency
    global_scaler = StandardScaler()
    global_scaler.fit(x_class)

    classDataa = []
    Dataa = []
    for _ in range(10):  # No need to vary seed in loop
        x_train, x_test, y_train, y_test = train_test_split(
            x_class, y_class, test_size=0.85, random_state=42
        )

        # Use the global scaler
        x_train = global_scaler.transform(x_train)
        x_test = global_scaler.transform(x_test)

        classData = {
            'x_train': x_train,
            'x_test': x_test,
            'y_train': y_train - 1,
            'y_test': y_test - 1
        }

        # Process image data with the same scaler
        sc = np.reshape(image, [image.shape[0] * image.shape[1], image.shape[2]])
        sc = global_scaler.transform(sc)
        scd = sc[label == 0]
        sc = np.reshape(sc, [image.shape[0], image.shape[1], image.shape[2]])

        Data = {'scd': scd, 'sc': sc, 'gtd': gtd}

        classDataa.append(classData)
        Dataa.append(Data)

    print('\nScene: ', sc.shape)
    print('\nClassification:')
    print('Training samples: ', len(classData['x_train']))
    print('Test samples: ', len(classData['x_test']))
    print('\nNumber of bands: ', str(classData['x_train'].shape[-1]))

    return classDataa, Dataa
def plotBands(selected_bands , data ,i, all_bands ):
    mean_reflectance = np.mean(data, axis=0)
    std_reflectance = np.std(data, axis=0)
    selected_bands = all_bands[selected_bands]

# Define spread regions (like probability distributions)
    upper_bound = mean_reflectance + std_reflectance
    lower_bound = mean_reflectance - std_reflectance

    plt.figure(figsize=(10, 5))
    plt.plot(bands, mean_reflectance, label="Mean Reflectance", color="blue", linewidth=2)
    plt.fill_between(bands, lower_bound, upper_bound, color="grey", alpha=0.2, label="±1 Std Dev")
    
    # Add vertical dashed lines & annotate selected bands
    for idx, sb in enumerate(selected_bands):
        plt.axvline(sb, color="red", linestyle="dashed", linewidth=1)
        plt.text(sb, upper_bound.max(), f"Band {selected_band_indices[idx]}", 
                 color="red", fontsize=10, rotation=0, ha="center", va="bottom", fontweight="bold")
    
    # Formatting
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Reflectance")
    plt.title("Spectral Band Reflectance Distribution")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    
    # Save and display the plot
    plot_path = "reflectance_distribution_(i).png"
    plt.savefig(plot_path)
    plt.show()
    
    plt.close()
    return plot_path 
   

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
        model_name , hyperparams ,  model = networks.SLRol(n_bands = n_bands, q = q)
        

        checkpoint_osen = tf.keras.callbacks.ModelCheckpoint(
            weightName, monitor='val_loss', verbose=0,
            save_best_only=True, mode='min', save_weights_only=True)

        callbacks_osen = [checkpoint_osen]

        if weights == 'False':
            mlflow.tensorflow.autolog()
            run_name = f"{model_name}_run{i}"
            with mlflow.start_run(run_name=run_name) as parent_run:
                
                parent_run_id = parent_run.info.run_id
                # # mlflow.log_param("CPU", cpu_info)
                # mlflow.log_param("CPU Cores", cpu_count)
                # mlflow.log_param("RAM (GB)", ram_total)
                # mlflow.log_param("OS", platform.system() + " " + platform.release())
                # mlflow.log_param("Python Version", platform.python_version())
                # mlflow.log_param("Scikit-Learn Version", sklearn.__version__)
                with open("run_id.txt", "w") as f:
                    f.write(parent_run_id)
                model_json = model.to_json()
                with open("model_architecture.json", "w") as json_file:
                    json_file.write(model_json)

                # Log model architecture
                mlflow.log_artifact("model_architecture.json")
            
                # Log custom hyperparameters
                mlflow.log_params(hyperparams)

                
                # mlflow.log_param("num_conv_layers", num_conv_layers)
                # mlflow.log_param("activation", activation)
                # mlflow.log_param("lambda_l1", lambda_l1)
                start_time=time.time()
                model.fit(xx, xx, batch_size = batchSize,
                        callbacks=callbacks_osen, shuffle=True,
                        validation_data=(xx, xx), epochs = epochs)
                execution_time = round(time.time() - start_time, 2)  # Seconds
                model_size = round(os.path.getsize(f"weights/Indian_pines_corrected/SRL-SOA_q3_run{i}.weights.h5") / (1024 ** 2), 2)
                
                
                mlflow.tensorflow.log_model(model,model_name)
                mlflow.log_metric("Execution_Time_seconds", execution_time)
                mlflow.log_param("Model_Size_MB", model_size)
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
                plotted_graph_path=plotBands(ind_a, xx,i, all_bands )
                mlflow.log_artifact(plotted_graph_path)
            
            # Create a DataFrame for the selected bands and their presence
                df_bands = pd.DataFrame([band_presence], columns=all_bands)
                # df_bands.to_csv()
                if os.path.exists( csv_file_path):
                    with open(csv_file_path, "a") as f:
                        # for i, cm_df in enumerate(confusion_matrices):
                            # f.write(f"Selected_bands - Run {i+1}\n")
                            # f.write(str(all_bands[:]))
                            df_bands.to_csv(f)
                            # f.write("\n")  # Add a newline between matrices
                else :
                    with open( csv_file_path, "w") as f:
                        # f.write(f"Selected_bands - Run {i+1}\n")
                        f.write(str(all_bands[:]))
                        df_bands.to_csv(f)
                        # f.write("\n")
        
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

    return list(ind_a)

def evalPerformance(classData, y_predict,n):
     # Number of runs
    print(f"The model shall evaluate for {n} times")

    # Initialize arrays for storing accuracy scores
    oa = np.zeros(n, dtype='float64')
    aa = np.zeros(n, dtype='float64')
    kappa = np.zeros(n, dtype='float64')
    run_id=None
    # Read the saved run ID from the training script
    with open("run_id.txt", "r") as f:
        run_id = f.read().strip()

    results = []
    confusion_matrices = []  # Store confusion matrices for CSV
    
    for i in range(n):
      with mlflow.start_run(run_id=run_id ,nested=True): 
        y_test = classData[i]['y_test']
        cm = confusion_matrix(y_test, y_predict[i])

        oa[i] = np.sum(y_test == y_predict[i]) / len(y_predict[i])
        aa[i] = balanced_accuracy_score(y_test, y_predict[i])
        kappa[i] = cohen_kappa_score(y_test, y_predict[i])

        results.append([i + 1, oa[i], aa[i], kappa[i]])
        report = classification_report(y_test, y_predict[i], output_dict=True)

        # Log SVM metrics
        mlflow.log_metric("svm_oa", oa[i], step=i)
        mlflow.log_metric("svm_aa", aa[i], step=i)
        mlflow.log_metric("svm_kappa", kappa[i], step=i)

        
        mlflow.log_dict(report, "svm_classification_report.json")

        # Save the SVM model
        # mlflow.sklearn.log_model(svm_model, "svm_model")
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
        plt.show()
        plot_path = f"cm_heatmap_run{i}.png"
        plt.savefig(plot_path)
        
        mlflow.log_artifact(plot_path)

    
        # display(plt.gcf())  # Display figure explicitly
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
        f.close()
        if i==6 :
            mlflow.log_artifact(cm_csv_path)
            mlflow.log_artifact(csv_path)
            
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
   
    with mlflow.start_run(run_id=run_id): 
        if n==6:
            avg_oa = np.mean(oa[:-1])
            avg_aa = np.mean(aa[:-1])
            avg_kappa = np.mean(kappa[:-1])
            mlflow.log_metric("all_bands_OA", oa[-1])
            mlflow.log_metric("all_bands_OA", aa[-1])
            mlflow.log_metric("all_KAPPA", kappa[-1])
        else:
            avg_oa = np.mean(oa)
            avg_aa = np.mean(aa)
            avg_kappa = np.mean(kappa)
            
            
    
        mlflow.log_metric("ModelAvg_OA", avg_oa)
        mlflow.log_metric("ModelAvg_AA", avg_aa)
        mlflow.log_metric("FinalAvg_KAPPA", avg_kappa)
        
        print(f"\nAverage Performance Over {n} Runs:")
        print(f"Overall Accuracy: {avg_oa:.4f}")
        print(f"Average Accuracy: {avg_aa:.4f}")
        print(f"Kappa Coefficient: {avg_kappa:.4f}")
    
        # Visualization: Plot different accuracy metrics across runs
        sns.set_style("whitegrid")
        plt.figure(figsize=(8, 5))
    
        x_labels = [f"Run {i+1}" for i in range(n)]
        width = 0.2  # Bar width



# Plot lines with markers
        plt.plot(np.arange(n), oa, marker="o", linestyle="-", color="blue", label="Overall Accuracy")
        plt.plot(np.arange(n), aa, marker="s", linestyle="-", color="green", label="Average Accuracy")
        plt.plot(np.arange(n), kappa, marker="^", linestyle="-", color="red", label="Kappa Coefficient")
        
        # Annotate each point with its value
        for i in range(n):
            plt.text(i, oa[i], f"{oa[i]:.2f}", ha="center", va="bottom", fontsize=10, color="blue")
            plt.text(i, aa[i], f"{aa[i]:.2f}", ha="center", va="bottom", fontsize=10, color="green")
            plt.text(i, kappa[i], f"{kappa[i]:.2f}", ha="center", va="bottom", fontsize=10, color="red")
        
        # Add dashed horizontal lines for averages
        plt.axhline(avg_oa, color="blue", linestyle="dashed", linewidth=1)
        plt.axhline(avg_aa, color="green", linestyle="dashed", linewidth=1)
        plt.axhline(avg_kappa, color="red", linestyle="dashed", linewidth=1)
        
        # Annotate the average lines
        plt.text(n - 1, avg_oa, f"Avg OA: {avg_oa:.2f}", color="blue", va="bottom", ha="right", fontsize=10)
        plt.text(n - 1, avg_aa, f"Avg AA: {avg_aa:.2f}", color="green", va="bottom", ha="right", fontsize=10)
        plt.text(n - 1, avg_kappa, f"Avg Kappa: {avg_kappa:.2f}", color="red", va="bottom", ha="right", fontsize=10)
        
        # Formatting
        plt.xticks(np.arange(n), x_labels)
        plt.ylabel("Score")
        plt.title("Performance Metrics Across Runs")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        
        # Save and log the plot
        plot_path = "performance_metrics.png"
        plt.savefig(plot_path)
        plt.savefig(plot_path)
        mlflow.log_artifacts(plot_path)
        
        plt.show()
        
        plt.close()
        



        
    
       
    # display(plt.gcf())  # Explicitly display the figure
  # Close the figure to prevent file access issues
