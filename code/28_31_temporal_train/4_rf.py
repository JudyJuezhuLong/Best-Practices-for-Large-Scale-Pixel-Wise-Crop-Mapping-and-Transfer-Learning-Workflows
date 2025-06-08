
#%%
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,precision_score,recall_score,cohen_kappa_score,f1_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import time
import pickle
import os
import json
import pandas as pd
import re

def _assert_suffix_match(suffix, path):
    assert re.search(r"\.{}$".format(suffix), path), "suffix mismatch"

def make_parent_dir(filepath):
    parent_path = os.path.dirname(filepath)
    if not os.path.isdir(parent_path):
        try:
            os.mkdir(parent_path)
        except FileNotFoundError:
            make_parent_dir(parent_path)
            os.mkdir(parent_path)
        print("[INFO] Make new directory: '{}'".format(parent_path))

def save_to_pkl(data, path):
    _assert_suffix_match("pkl", path)
    make_parent_dir(path)
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=-1)
    print("[INFO] Save as pkl: '{}'".format(path))

def load_from_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def save_to_pth_rf(data, path, model=True):
    _assert_suffix_match("pth", path)
    make_parent_dir(path)
    if model:
        with open(path, 'wb') as file:
            pickle.dump(data, file)
    print("[INFO] Save as pth: '{}'".format(path))

def load_from_pth_rf(path):
    with open(path, "rb") as f:
        return pickle.load(f)
    
#%%
yy='2022'
preprocessing_method='7days_ln'
model_type='RF'
sample_size='400k'
target_scene_id = '28_31'


with open("../sample_data/input_path_config.json", "r") as f:
    file_paths = json.load(f)

x_dir_key = 'x_data_dir_'+target_scene_id+'_'+preprocessing_method+'_'+yy
y_dir_key = 'y_data_dir_'+target_scene_id+'_'+preprocessing_method+'_'+yy
print("X_data path:",file_paths[x_dir_key])
print("label path:",file_paths[y_dir_key])
x_data = np.load(file_paths[x_dir_key])
y_data = np.load(file_paths[y_dir_key]).ravel()
print("X_data shape",x_data.shape)


    
# Check for all NaN values in the second dimension
any_nan_in_second_dim = np.any(np.isnan(x_data), axis=(1, 2))

# Filter out the records where the second dimension contains all NaNs
x_data = x_data[~any_nan_in_second_dim]
y_data = y_data[~any_nan_in_second_dim]
print(x_data.shape)
x_data = x_data.reshape(x_data.shape[0],-1)
print(x_data.shape)


le = preprocessing.LabelEncoder()
le.fit(y_data)

# print(le.classes_)
y_data = le.transform(y_data)

# Filter out NaNs using the boolean mask
x_data[np.isnan(x_data)] = 0

# Select 10,000 samples randomly from the dataset
sample_size = 400000 # Be used to select the pre-trained model
sample_size_target = 4000 # Be used to select the Test set
train_size_target = 9000 # Be used to select the Train set and Valid set (9000 in total)

# select the Test set
np.random.seed(42) # Keep the test set is the same with the directly applying pre-trained models
random_indices = np.random.choice(len(x_data), size=sample_size_target, replace=False)
x_test = x_data[random_indices]
y_test = y_data[random_indices]
print("x_test.shape",x_test.shape, 'y_test',y_test.shape)
x_train_val = x_data[~np.isin(np.arange(len(x_data)), random_indices)]
y_train_val = y_data[~np.isin(np.arange(len(y_data)), random_indices)]
print("x_train_val.shape",x_train_val.shape, 'y_train_val',y_train_val.shape)

# select the Train set and Valid set
x_train_val, _, y_train_val, _ = train_test_split(x_train_val, y_train_val, stratify= y_train_val, train_size=train_size_target, random_state=42)
# x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=1000, random_state=42)

# Number of folds
n_folds = 9

# Initialize KFold
kf = KFold(n_splits=n_folds, shuffle=True, random_state=18)

# Get all fold indices for further splitting
folds = list(kf.split(x_train_val))

# Iterate over each fold
# Loop over each fold
for fold_i in range(n_folds): 
    # Validation set: take the first fold from remaining folds
    val_idx = folds[fold_i][1]
    
    # Remaining folds: the ones that are not the test fold
    remaining_folds = [folds[j][1] for j in range(n_folds) if j != fold_i]  # Use folds[j][1]
    
    # Training set: concatenate the remaining 8 folds
    train_idx = np.concatenate(remaining_folds)
    
    # Split data into train, validation, and test sets
    x_train, y_train = x_train_val[train_idx], y_train_val[train_idx]
    x_val, y_val = x_train_val[val_idx], y_train_val[val_idx]
    # Combine training and validation data for scaling
    x_train_val = np.concatenate((x_train, x_val))
    y_train_val = np.concatenate((y_train, y_val))
    
    # Print sizes for each split (you can replace this with model training)
    print(f"Experiment {fold_i+1}:")
    print(f"Train size: {x_train.shape}, Validation size: {x_val.shape}, Test size: {x_test.shape}")

    sc = StandardScaler()
    sc.fit(x_train_val)
    x_train_std = sc.transform(x_train)
    x_test_std = sc.transform(x_test)
    save_to_pkl(sc, os.path.join(f'./output_Anonymous_{fold_i}', f"{yy}_{preprocessing_method}_{model_type}_{sample_size}_scaler.pkl"))
    
    start_time = time.time()
    rf = RandomForestClassifier(random_state=100,n_estimators=500,max_features=8,n_jobs=-1)
    rf.fit(x_train_std, y_train)
    end_time = time.time()
    training_time = end_time - start_time
    print("training_time",training_time)
    
    os.makedirs(f'./output_Anonymous_{fold_i}', exist_ok=True)
    save_to_pth_rf(rf, os.path.join(f'./output_Anonymous_{fold_i}', f"{yy}_{preprocessing_method}_{model_type}_{sample_size}_best.pth"))
    
    prediction = rf.predict(x_test_std)
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
    np.savetxt(os.path.join(f'./output_Anonymous_{fold_i}', f"{yy}_{preprocessing_method}_{model_type}_{sample_size}_confusion_matrix.csv"), confusion_matrix0, delimiter=",")

    # Create a DataFrame for metrics
    metrics_df = pd.DataFrame({
        "Preprocessing": [preprocessing_method],
        "Model": [model_type],
        "Train Size": [x_train.shape[0]],
        "Test Size": [x_test.shape[0]], 
        "Validation Size": [x_val.shape[0]],
        "Overall Accuracy": [overall_accuracy],
        "Class 0 Accuracy": [acc_for_each_class[0]],
        "Class 1 Accuracy": [acc_for_each_class[1]],
        "Class 2 Accuracy": [acc_for_each_class[2]],
        "Recall": [recall0],
        "Precision": [precision0],
        "Kappa": [kappa],
        "F1 Score": [f_score],
        "Train Time":[training_time]
    })

    # Save the metrics table to CSV
    metrics_df.to_csv(os.path.join(f'./output_Anonymous_{fold_i}', f"{yy}_{preprocessing_method}_{model_type}_{sample_size}_metrics.csv"), index=False)
