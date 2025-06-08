
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
from sklearn.model_selection import KFold
import tqdm
import math 
import json
import pandas as pd
gpu_ids = os.environ.get("CUDA_VISIBLE_DEVICES", "0")  # Get assigned GPUs
gpu_list = [int(i) for i in gpu_ids.split(",")]
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
yy='2023'
preprocessing_method='7days_ln_pheno_peak'
model_type='transformer'
sample_size='630k'
scene_id = '27_30'
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
    
# class TransformerModel(nn.Module):
#     def __init__(self, seed=0,input_dim=6, num_classes=3, d_model=512, n_head=8, n_layers=2,
#                  d_inner=128, activation="relu", dropout=0.01):

#         super(TransformerModel, self).__init__()
#         self.modelname = f"TransformerEncoder_input-dim={input_dim}_num-classes={num_classes}_" \
#                          f"d-model={d_model}_d-inner={d_inner}_n-layers={n_layers}_n-head={n_head}_" \
#                          f"dropout={dropout}"

#         encoder_layer = TransformerEncoderLayer(d_model, n_head, d_inner, dropout, activation)
#         encoder_norm = LayerNorm(d_model)

#         self.inlinear = Linear(input_dim, d_model)
#         self.relu = ReLU()
#         self.transformerencoder = TransformerEncoder(encoder_layer, n_layers, encoder_norm)
#         self.flatten = Flatten()
#         self.outlinear = Linear(d_model, num_classes)

#     def forward(self,x):
#         x = self.inlinear(x)
#         x = self.relu(x)
#         x = x.transpose(0, 1) # N x T x D -> T x N x D
#         x = self.transformerencoder(x)
#         x = x.transpose(0, 1) # T x N x D -> N x T x D
#         x = x.max(1)[0]
#         x = self.relu(x)
#         logits = self.outlinear(x)

#         logprobabilities = F.log_softmax(logits, dim=-1)
#         return logprobabilities, logits

# class Flatten(nn.Module):
#     def forward(self, input):
#         return input.reshape(input.size(0), -1)
    
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
    
def make_data_loader(x, y,  shuffle,batch_size,num_workers,_collate_fn,drop_last):
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

def save_to_pkl(data, path):
    _assert_suffix_match("pkl", path)
    make_parent_dir(path)
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=-1)
    print("[INFO] Save as pkl: '{}'".format(path))


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

def drawAccuracyCurves_(cls_accs_ls,out_path):
    pd.DataFrame(cls_accs_ls).to_csv(out_path+'_cls_accs.csv', index=False)
    fig, ax = plt.subplots( nrows=1, ncols=1 )
    ax.plot(cls_accs_ls, label='Source domain training accuracy')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0.7,1)
    ax.legend()
    fig.savefig(out_path)
    plt.close(fig)


def drawLossCurves(loss_ls,out_path):
    pd.DataFrame(loss_ls).to_csv(out_path+'_loss.csv', index=False)
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

def _eval_perf(net, dataloader, best_loss, device):
    # Validation phase
    net.eval()
    val_loss = 0
    val_correct = 0
    with torch.no_grad():
        for j, s in enumerate(dataloader):
            x_s = s["x"].to(device)
            labels_s = s["y"].to(device)
            y_s, f = net(x_s)
            val_loss += criterion(y_s, labels_s).item()
            yt_pred = torch.max(y_s, dim=1)[1]
            val_correct += (yt_pred == labels_s).sum().item()
    
    val_loss /= len(val_loader)
    val_correct /= len(val_loader)
    # print(f"Epoch {epoch + 1}, Validation Loss: {val_loss:.4f}")

    # Save checkpoint if validation loss improves
    if val_loss < best_loss:
        best_loss = val_loss
        save_checkpoint(net, optimizer, epoch, val_loss, os.path.join(f'./output_Anonymous_{fold_i}', f"{yy}_{preprocessing_method}_{model_type}_{sample_size}_best.pth"))
        # save_checkpoint(classifier, optimizer, epoch, val_loss, 'model_checkpoint.pth')
    return val_loss, val_correct, best_loss
    
# import breizhcrops
# from breizhcrops import BreizhCrops
#%%
    
with open("../sample_data/input_path_config.json", "r") as f:
    file_paths = json.load(f)

x_dir_key = 'x_data_dir_'+scene_id+'_'+preprocessing_method+'_CDL'
y_dir_key = 'y_data_dir_'+scene_id+'_'+preprocessing_method+'_CDL'
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

# Select 400,000 samples randomly from the dataset
sample_size = 400000
random_indices = np.random.choice(len(x_data), size=sample_size, replace=False)

x_data = x_data[random_indices]
y_data = y_data[random_indices]

# Number of folds
n_folds = 10

# Initialize KFold
kf = KFold(n_splits=n_folds, shuffle=True, random_state=18)

# Get all fold indices for further splitting
folds = list(kf.split(x_data))

# Loop over each fold
for fold_i in range(n_folds): 

    # Test set: the second element in each tuple is the test indices
    test_idx = folds[fold_i][1]
    
    # Remaining folds: the ones that are not the test fold
    remaining_folds = [folds[j][1] for j in range(n_folds) if j != fold_i]  # Use folds[j][1]
    
    # Validation set: take the first fold from remaining folds
    val_idx = remaining_folds[0]
    
    # Training set: concatenate the remaining 8 folds
    train_idx = np.concatenate(remaining_folds[1:])
    
    # Split data into train, validation, and test sets
    x_train, y_train = x_data[train_idx], y_data[train_idx]
    x_val, y_val = x_data[val_idx], y_data[val_idx]
    x_test, y_test = x_data[test_idx], y_data[test_idx]
    # Combine training and validation data for scaling
    x_train_val = np.concatenate((x_train, x_val))
    y_train_val = np.concatenate((y_train, y_val))
    
    # Print sizes for each split (you can replace this with model training)
    print(f"Experiment {fold_i+1}:")
    print(f"Train size: {x_train.shape}, Validation size: {x_val.shape}, Test size: {x_test.shape}")

    batch_size=int(pow(2, 12))
        
    # y_train_val = le.transform(y_train_val)
    # y_test = le.transform(y_test)
    # print("num_classes",num_classes)
    scaler, x_train_val, x_test =normalize_without_scaler(x_train_val, x_test)
    x_train = normalize_with_scaler(scaler,x_train)
    x_val = normalize_with_scaler(scaler,x_val)
    save_to_pkl(scaler, os.path.join(f'./output_Anonymous_{fold_i}', f"{yy}_{preprocessing_method}_{sample_size}_scaler.pkl"))
    

    """
    num_workers=2
    drop_last=False
    shuffle=False
    _collate_fn=_collate_fn
    device = torch.device(f"cuda:{gpu_list[0]}" if torch.cuda.is_available() else "cpu")
    print(f"Using GPU(s): {gpu_list}")
    # device = torch.device("cuda:0")  




    train_loader = make_data_loader(x_train, y_train,  shuffle=True,batch_size=batch_size,num_workers=num_workers,_collate_fn=_collate_fn,drop_last=True)# =train_dataloader
    # train_loader = make_data_loader(x_train_val, y_train_val, shuffle=True,batch_size=batch_size,num_workers=num_workers,_collate_fn=_collate_fn,drop_last=True)# =train_dataloader
    val_loader = make_data_loader(x_val, y_val, shuffle=True,batch_size=batch_size,num_workers=num_workers,_collate_fn=_collate_fn,drop_last=True)# =train_dataloader
    test_loader = make_data_loader(x_test, y_test, shuffle=False,batch_size=batch_size,num_workers=num_workers,_collate_fn=_collate_fn,drop_last=False)# =test_dataloader

    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                        'This will turn on the CUDNN deterministic setting, '
                        'which can slow down your training considerably! '
                        'You may see unexpected behavior when restarting '
                        'from checkpoints.')
    
    transformer = TransformerModel(seq_len=x_train.shape[1])
    lr=0.0005
    weight_decay=0.001
    mode="min"
    factor=0.1
    patience=10
    verbose=True
    classifier = transformer.to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.0005)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode=mode, factor=factor,
        patience=patience, verbose=verbose)
    criterion = torch.nn.CrossEntropyLoss()
    start_time = time.time()
    end_time = time.time()
    best_loss = float('inf')

    def load_checkpoint(PATH):
        checkpoint = torch.load(PATH)
        classifier.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # epoch = checkpoint['epoch']
        # loss = checkpoint['loss']
        return classifier,optimizer
    checkpoint_path = os.path.join(f'./output_Anonymous_{fold_i}', f"{yy}_{preprocessing_method}_{model_type}_{sample_size}_best.pth")
    try:
        classifier,optimizer = load_checkpoint(checkpoint_path)
        print('Using the best check point ...')
    except:
        pass

    cls_accs_ls = []
    loss_ls = []
    for epoch in range(100):
        classifier.train()
        for idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            X = batch['x'].to(device)
            y = batch['y'].to(device)
            # field_ids = batch['field_ids']

            #   X,y,field_ids = batch
            y_pred, _ = classifier(X)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            end_time = time.time()
        lr_scheduler.step(loss.item())

        val_loss, val_correct, best_loss = _eval_perf(classifier, val_loader, best_loss, device)
        # cls_accs_ls.append(round(cls_accs.val,2)*0.01)
        # loss_ls.append(round(losses.val,2))
        cls_accs_ls.append(val_correct)
        loss_ls.append(val_loss)
        
    training_time = end_time - start_time
    print("training_time",training_time)

    save_to_pth(classifier, os.path.join(f'./output_Anonymous_{fold_i}', f"{yy}_{preprocessing_method}_{model_type}_{sample_size}.pth"))
    drawAccuracyCurves_(cls_accs_ls, os.path.join(f'./output_Anonymous_{fold_i}', f"{yy}_{preprocessing_method}_{model_type}_{sample_size}_acc.png"))
    drawLossCurves(loss_ls, os.path.join(f'./output_Anonymous_{fold_i}', f"{yy}_{preprocessing_method}_{model_type}_{sample_size}_loss.png"))

    #%%
    # field_ids = np.array(range(len(y_target)))
    # test_dataloader = make_data_loader(x_target, y_target, shuffle=False,batch_size=batch_size,num_workers=num_workers,_collate_fn=_collate_fn,drop_last=False)

    # prediction, _ = classifier(train_target_loader)
    yt_soft_pred_batch_list = []
    classifier,optimizer = load_checkpoint(checkpoint_path)
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
    """