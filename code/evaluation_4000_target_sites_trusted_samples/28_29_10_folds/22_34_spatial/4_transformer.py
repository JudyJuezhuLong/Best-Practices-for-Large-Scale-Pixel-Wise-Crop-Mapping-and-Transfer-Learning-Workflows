
#%%
# ! pip install tqdm
#%%
import torch
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer
from torch.nn.modules import LayerNorm, Linear, Sequential, ReLU
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F
import time
import warnings
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
# from models_DCM.classifier import Classifier
import numpy as np
from sklearn import preprocessing
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import re
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,precision_score,recall_score,cohen_kappa_score,f1_score
import tqdm
import json
import math
import pandas as pd

# Automatically detect allocated GPUs
gpu_ids = os.environ.get("CUDA_VISIBLE_DEVICES", "0")  # Get assigned GPUs
gpu_list = [int(i) for i in gpu_ids.split(",")]
yy='2023'
preprocessing_method='7days_ln'
model_type='transformer'
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


#%%
__all__ = ['TransformerModel']
seed = None
class TransformerModel(nn.Module):
    def __init__(
        self, seed=0, pe_tau = 10000, input_dim = 6, d_model = 512, nhead = 8,
        dim_feedforward = 256, dropout = 0.2, num_layers = 2, seq_len = None, num_classes = 3
    ):
        super().__init__()
        self._set_reproducible(seed)
        
        self.fc1 = nn.Linear(
            in_features=input_dim, out_features=d_model
        )  # increase dimension
        self.pos_encoding = PositionalEncoding(
            d_model=d_model, pe_tau=pe_tau, max_seq_len=seq_len
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout
        )
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers, encoder_norm
        )
        self.pool = nn.AvgPool1d(seq_len)
        
        self.fc2 = nn.Linear(
            in_features=d_model,
            out_features=num_classes,
        )
    
    def _set_reproducible(self, seed, cudnn=False):
        np.random.seed(seed)
        torch.manual_seed(seed)
        if cudnn:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        # => (seq_len, batch, input_dim)
        x = x.permute((1, 0, 2))
        # fc1_out: (seq_len, batch, d_model)
        fc1_out = self.fc1(x)
        # encoder_in: (seq_len, batch, d_model)
        encoder_in = self.pos_encoding(fc1_out)
        # encoder_out: (seq_len, batch, d_model)
        encoder_out = self.encoder(encoder_in)
        # pool_out: (batch, d_model, 1)
        pool_out = self.pool(encoder_out.permute((1, 2, 0)))
        # outputs: (batch, num_classes)
        logits = self.fc2(pool_out.squeeze())
        logprobabilities = F.log_softmax(logits, dim=-1)
        return logprobabilities, logits
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, pe_tau, max_seq_len=5000):
        super().__init__()

        # pe: positional encoding matrix
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).float().unsqueeze(1)
        divisor = -torch.exp(
            torch.arange(0, d_model, 2).float()
            * math.log(pe_tau) / d_model
        )
        pe[:, 0::2] = torch.sin(position * divisor)
        pe[:, 1::2] = torch.cos(position * divisor)
        # pe: (max_seq_len, d_model) => (max_seq_len, 1, d_model)
        pe = pe.unsqueeze(0).permute((1, 0, 2))
        self.register_buffer("pe", pe)

    def forward(self, x):
        # (sql_len, batch, d_model) + (sql_len, 1, d_model)
        # => (sql_len, batch, d_model)
        return x + self.pe[:x.shape[0], :, :]
    
#%%

def define_Fsize_and_Nclass(x,y):
    input_feature_size = x.shape[2]
    num_classes = np.unique(y).shape[0]
    return input_feature_size,num_classes

def normalize_without_scaler(x_train, x_test):
    scaler = torch.nn.BatchNorm1d(
        x_train.shape[2], eps=0, momentum=1, affine=False
    )
    scaler.train()
    x_train = scaler(
        torch.FloatTensor(x_train.transpose((0, 2, 1)))
    ).numpy().transpose((0, 2, 1))
    scaler.eval()
    x_test = scaler(
        torch.FloatTensor(x_test.transpose((0, 2, 1)))
    ).numpy().transpose((0, 2, 1))
    return scaler, x_train, x_test

def normalize_with_scaler(scaler, x_test):
    x_test = scaler(
        torch.FloatTensor(x_test.transpose((0, 2, 1)))
    ).numpy().transpose((0, 2, 1))
    return x_test
    
def make_data_loader(x,y,shuffle,batch_size,num_workers,_collate_fn,drop_last):
    return DataLoader(
        CropMappingDataset(x, y),
        batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, collate_fn=_collate_fn,
        drop_last=drop_last
    )

def _collate_fn(batch):
    """
    define how to aggregate samples to batch
    """
    return {
        "x": torch.FloatTensor(
            np.array([sample["sample_x"] for sample in batch])
        ),
        "y": torch.LongTensor(
            np.array([sample["sample_y"] for sample in batch])
        )
    }
    

class CropMappingDataset(Dataset):
    """
    crop classification dataset
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return {"sample_x": self.x[idx], "sample_y": self.y[idx]}  

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

def save_to_pth(data, path, model=True):
    _assert_suffix_match("pth", path)
    make_parent_dir(path)
    if model:
        if hasattr(data, "module"):
            data = data.module.state_dict()
        else:
            data = data.state_dict()
    torch.save(data, path)
    print("[INFO] Save as pth: '{}'".format(path))

def load_from_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)
    
def drawAccuracyCurves_(cls_accs_ls,out_path):
    fig, ax = plt.subplots( nrows=1, ncols=1 )
    ax.plot(cls_accs_ls, label='Source domain training accuracy')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0.7,1)
    ax.legend()
    fig.savefig(out_path)
    plt.close(fig)


def drawLossCurves(loss_ls,out_path):
    fig, ax = plt.subplots( nrows=1, ncols=1 )
    ax.plot(loss_ls, label='Training loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    fig.savefig(out_path)
    plt.close(fig)

# Define a function to save the checkpoint
def save_checkpoint(model, optimizer, epoch, loss, filepath):
    _assert_suffix_match("pth", filepath)
    make_parent_dir(filepath)
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(state, filepath)
    print(f"Checkpoint saved at epoch {epoch} with loss {loss:.4f}")


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


    
# Check for all NaN values in the second dimension
any_nan_in_second_dim = np.any(np.isnan(x_data), axis=(1, 2))

# Filter out the records where the second dimension contains all NaNs
x_data = x_data[~any_nan_in_second_dim]
y_data = y_data[~any_nan_in_second_dim]
print(x_data.shape)
# print(np.unique(np.isnan(x_data),return_counts=True))


le = preprocessing.LabelEncoder()
le.fit(y_data)

# print(le.classes_)
y_data = le.transform(y_data)

# Filter out NaNs using the boolean mask
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
for fold_i in range(n_folds): 
    
    # Print sizes for each split (you can replace this with model training)
    print(f"Experiment {fold_i+1}:")
    # print(f"Train_Val size: {x_train_val.shape}, Test size: {x_test.shape}")

    input_feature_size,num_classes=define_Fsize_and_Nclass(x_test,y_test)
    PATH = os.path.join(f'{best_models_directory1}/output_Anonymous_{fold_i}', f"{yy}_{preprocessing_method}_{sample_size}_scaler.pkl")
    scaler = load_from_pkl(PATH)
    print("[INFO] Read pkl: '{}'".format(PATH))
    x_test_scale = normalize_with_scaler(scaler, x_test)


    num_workers=2
    _collate_fn=_collate_fn
    device = torch.device(f"cuda:{gpu_list[0]}" if torch.cuda.is_available() else "cpu") 
    batch_size=32

    test_loader = make_data_loader(x_test_scale, y_test, shuffle=False,batch_size=batch_size,num_workers=num_workers,_collate_fn=_collate_fn,drop_last=False)# =test_dataloader

    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                        'This will turn on the CUDNN deterministic setting, '
                        'which can slow down your training considerably! '
                        'You may see unexpected behavior when restarting '
                        'from checkpoints.')
    #%%
    transformer = TransformerModel(seq_len=x_test.shape[1])
    lr=0.0001

    classifier = transformer.to(device)
    
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
    
    def load_checkpoint(PATH):
        checkpoint = torch.load(PATH)
        classifier.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # epoch = checkpoint['epoch']
        # loss = checkpoint['loss']
        return classifier,optimizer
    checkpoint_path = os.path.join(f'{best_models_directory1}/output_Anonymous_{fold_i}', f"{yy}_{preprocessing_method}_{model_type}_{sample_size}_best.pth")
    # print('best_models_directory1',best_models_directory1)
    # checkpoint_path = os.path.join(f'{best_models_directory1}/output_Anonymous_{fold_i}', "2023_7days_ln_transformer_400000_best.pth")
    try:
        classifier,optimizer = load_checkpoint(checkpoint_path)
        print(f'Using the best check point ...{checkpoint_path}')
    except:
        pass


    yt_soft_pred_batch_list = []

    with torch.no_grad():
        for j, s in enumerate(test_loader):
            x_s = s["x"].to(device)
            labels_s = s["y"].to(device)
            optimizer.zero_grad()
            y_pred, _ = classifier(x_s)
            yt_soft_pred_batch_list.append(F.softmax(y_pred, dim=1))
        y_soft_pred = torch.cat(
                yt_soft_pred_batch_list, dim=0
            ).cpu().numpy()
        prediction = np.argmax(y_soft_pred, axis=1)

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
