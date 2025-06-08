#%%
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,precision_score,recall_score,cohen_kappa_score,f1_score
from sklearn.model_selection import train_test_split
import time
import pickle
import joblib
import os
import json
import pandas as pd
from sklearn.model_selection import KFold
import torch
#%%
yy='2023'
preprocessing_method='7days_ln'
model_type='RF'
sample_size='400k'

target_sample_size = '180k'
source_scene_id = '28_29'
target_scene_id = '22_34'
scene_id = source_scene_id+'_'+target_scene_id
current_directory = os.path.dirname(os.path.abspath(__file__))
print(current_directory)
parent_l3_directory = os.path.dirname(current_directory)
parent_l2_directory = os.path.dirname(parent_l3_directory)
parent_l1_directory = os.path.dirname(parent_l2_directory)

best_models_directory1 = parent_l1_directory + current_directory[len(parent_l2_directory):len(parent_l3_directory)]

def load_from_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def load_checkpoint(PATH):
    loaded_clf = joblib.load(checkpoint_path)
    return loaded_clf
#%%
with open("/mnt/mridata/Anonymouslong/best_practice_pixel/sample_data/input_path_config.json", "r") as f:
    file_paths = json.load(f)

x_dir_key = 'x_data_dir_'+target_scene_id+'_'+preprocessing_method
y_dir_key = 'y_data_dir_'+target_scene_id+'_'+preprocessing_method
print("X_data path:",file_paths[x_dir_key])
print("label path:",file_paths[y_dir_key])
x_data = np.load(file_paths[x_dir_key])
y_data = np.load(file_paths[y_dir_key]).ravel()
print("X_data shape",x_data.shape)
 
#%%

# Check for all NaN values in the second dimension
any_nan_in_second_dim = np.any(np.isnan(x_data), axis=(1, 2))

# Filter out the records where the second dimension contains all NaNs
x_data = x_data[~any_nan_in_second_dim]
y_data = y_data[~any_nan_in_second_dim]
# print(x_data.shape)
# print(np.unique(np.isnan(x_data),return_counts=True))

#%%
print(x_data.shape)

le = preprocessing.LabelEncoder()
le.fit(y_data)

# print(le.classes_)
y_data = le.transform(y_data)


x_data[np.isnan(x_data)] = 0

# Select 10,000 samples randomly from the dataset
sample_size = 400000 # Be used to select the pre-trained model
sample_size_target = 4000 # Be used to select the Test set
# train_size_target = 2000 # Be used to select the Train set and Valid set (2000 in total)

# select the Test set
np.random.seed(42) # Keep the test set is the same with the directly applying pre-trained models
random_indices = np.random.choice(len(x_data), size=sample_size_target, replace=False)
x_test = x_data[random_indices]
y_test = y_data[random_indices]

# Number of folds
n_folds = 10
# # Loop over each fold
for fold_i in range(n_folds): 
    
    # Print sizes for each split (you can replace this with model training)
    print(f"Experiment {fold_i+1}:")

    PATH = os.path.join(f"{best_models_directory1}/output_Anonymous_{fold_i}", f"{yy}_{preprocessing_method}_{model_type}_{sample_size}_scaler.pkl")
    scaler = load_from_pkl(PATH)


    checkpoint_path = os.path.join(f'{best_models_directory1}/output_Anonymous_{fold_i}', f"{yy}_{preprocessing_method}_{model_type}_{sample_size}_best.pth")
    try:
        classifier = load_checkpoint(checkpoint_path)
        print(f'Using the best check point ...{checkpoint_path}')
    except:
        pass
    # checkpoint_path = os.path.join('./best_models', yy+"_"+preprocessing_method+"_"+model_type+"_"+sample_size+".pkl")

    # loaded_clf = load_checkpoint(checkpoint_path)

    x_test_reshape = x_test.reshape(x_test.shape[0],-1)
    x_test_scale = scaler.transform(x_test_reshape)
    prediction = classifier.predict(x_test_scale)

    # Compute metrics
    confusion_matrix0 = confusion_matrix(y_test, prediction)
    overall_accuracy = accuracy_score(y_test, prediction)
    acc_for_each_class = np.diag(confusion_matrix0) / np.bincount(y_test)
    recall0 = recall_score(y_test, prediction, average='macro')  # Use 'macro' for multi-class
    precision0 = precision_score(y_test, prediction, average='macro')  # Use 'macro' for multi-class
    kappa = cohen_kappa_score(y_test, prediction)
    f_score = f1_score(y_test, prediction, average='micro')
    print('confusion_matrix : \n', confusion_matrix0)
    print('acc_for_each_class : \n', acc_for_each_class)
    print('overall_accuracy: {0:f}'.format(overall_accuracy))
    print('recall0: {0:f}'.format(recall0))
    print('precision0: {0:f}'.format(precision0))
    print('kappa: {0:f}'.format(kappa))
    print('f_score: {0:f}'.format(f_score))

    # Export confusion matrix to CSV (without row and column names)
    os.makedirs(f'./output_Anonymous_{fold_i}', exist_ok=True)
    np.savetxt(os.path.join(f'./output_Anonymous_{fold_i}', f"{yy}_{preprocessing_method}_{model_type}_{sample_size}_confusion_matrix.csv"), confusion_matrix0, delimiter=",")

    # Create a DataFrame for metrics
    metrics_df = pd.DataFrame({
        "Preprocessing": [preprocessing_method],
        "Model": [model_type],
        "Train Size": 0,
        "Test Size": [x_test.shape[0]], 
        "Validation Size": 0,
        "Overall Accuracy": [overall_accuracy],
        "Class 0 Accuracy": [acc_for_each_class[0]],
        "Class 1 Accuracy": [acc_for_each_class[1]],
        "Class 2 Accuracy": [acc_for_each_class[2]],
        "Recall": [recall0],
        "Precision": [precision0],
        "Kappa": [kappa],
        "F1 Score": [f_score],
        "Train Time":0
    })

    # Save the metrics table to CSV
    metrics_df.to_csv(os.path.join(f'./output_Anonymous_{fold_i}', f"{yy}_{preprocessing_method}_{model_type}_{sample_size}_metrics.csv"), index=False)
