#%%
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,precision_score,accuracy_score,cohen_kappa_score,f1_score
from sklearn.model_selection import train_test_split
import time
import pickle
import os
import joblib
from sklearn.model_selection import KFold
import torch
import rasterio
import pandas as pd

#%%
yy='2023'
preprocessing_method='7days_ln'
model_type='RF'
sample_size='400000'
scene_id = ''

# Automatically detect allocated GPUs
gpu_ids = os.environ.get("CUDA_VISIBLE_DEVICES", "0")  # Get assigned GPUs
gpu_list = [int(i) for i in gpu_ids.split(",")]

current_directory = os.path.dirname(os.path.abspath(__file__))
print(current_directory)
parent_l3_directory = os.path.dirname(current_directory)
parent_l2_directory = os.path.dirname(parent_l3_directory)
parent_l1_directory = os.path.dirname(parent_l2_directory)

best_models_directory1 = parent_l1_directory + current_directory[len(parent_l2_directory):len(parent_l3_directory)]


fold_i = 8

def load_from_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def load_checkpoint(PATH):
    loaded_clf = joblib.load(PATH)
    return loaded_clf

def normalize_with_scaler(scaler, x_test):
    x_test = scaler(
        torch.FloatTensor(x_test.transpose((0, 2, 1)))
    ).numpy().transpose((0, 2, 1))
    return x_test

def createFolder(out_folder):
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    return True

def rescale_X_data(wrong_reflectance):
    correct_reflectance = (wrong_reflectance * 0.275) - 0.2
    return correct_reflectance


def pred_map(ts_stamp, tif_path):

    with rasterio.open(tif_path) as src:
        profile = src.profile
        height, width = src.height, src.width
        tif_read_data = src.read()  # Shape: (bands, height, width)
        x_test = tif_read_data.reshape(int(tif_read_data.shape[0]/6), 6, height, width).transpose(2, 3, 0, 1)
        x_test = x_test[:,:,16:39,:]
        x_test = x_test.reshape(height * width, 23, 6)

        x_test = rescale_X_data(x_test)

        print(x_test.shape)
        # plt.plot((reshaped_data[450,450,:,3]-reshaped_data[450,450,:,2])/(reshaped_data[450,450,:,3]+reshaped_data[450,450,:,2]))
        # plt.savefig('test_ts.png')

        mask = ~np.any(np.isnan(x_test), axis=(1,2))
        x_test = x_test[mask]
        print(x_test.shape)
        if x_test.shape[0]==0:
            return f'SKIP {ts_stamp}'

    # scaler_path = os.path.join('../pkl', preprocessing_method+"_400k_scaler.pkl")
    # scaler = pickle.load(open(scaler_path, 'rb'))
    # x_test = normalize_with_scaler(scaler, x_test)
    # x_test = x_test.reshape(x_test.shape[0],-1)
    # y_test = np.zeros(x_test.shape[0])

    PATH = os.path.join(f"{best_models_directory1}/output_Anonymous_{fold_i}", f"{yy}_{preprocessing_method}_{model_type}_{sample_size}_scaler.pkl")
    scaler = load_from_pkl(PATH)

    x_test = x_test.reshape(x_test.shape[0],-1)
    x_test = scaler.transform(x_test)



    checkpoint_path = os.path.join(f'{best_models_directory1}/output_Anonymous_{fold_i}', f"{yy}_{preprocessing_method}_{model_type}_{sample_size}_best.pth")
    try:
        classifier = load_checkpoint(checkpoint_path)
        print(f'Using the best check point ...{checkpoint_path}')
    except:
        pass

    start_time = time.time()
    prediction = classifier.predict(x_test)
    end_time = time.time()
    predicted_map = np.full(mask.shape, np.nan)
    predicted_map[mask] = prediction

    predicted_map_2d = predicted_map.reshape((height, width))
    # print('predicted_map_2d shape',predicted_map_2d.shape)

    predicting_time = end_time - start_time
    print('predicting_time',predicting_time)
    # Save the output as a new GeoTiff with the same profile
    # output_file = "predicted_map.tif"
    output_file = os.path.join(f'./{preprocessing_method}_{model_type}', f"{ts_stamp}.tif")
    profile.update(dtype=rasterio.float32, count=1)

    with rasterio.open(output_file, "w", **profile) as dst:
        dst.write(predicted_map_2d.astype(rasterio.float32), 1)
    # except:
    #     print(ts_stamp, 'failed')

createFolder(f'./{preprocessing_method}_{model_type}')
print(f'create Folder {preprocessing_method}_{model_type}')
# csv_path = '../csv_tif_list/28_29_'+preprocessing_method+'_tif_list.csv'
# file = pd.read_csv(csv_path)
# for i in range(file.shape[0]):
#     pred_map(file.iloc[i,:])

pred_map('02500_03500','./7day_ln_image/2023-0000002500-0000003500.tif')