
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
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,precision_score,accuracy_score,cohen_kappa_score,f1_score
import tqdm
import pandas as pd
import re
import rasterio
import math

yy='2023'
preprocessing_method='7days_ln'
model_type='transformer'
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

fold_i = 9
batch_size=int(pow(2, 12))

#%%
__all__ = ['TransformerModel']
seed = 0

def createFolder(out_folder):
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    return True

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
    
def make_data_loader(x, y, field_ids, shuffle,batch_size,num_workers,_collate_fn,drop_last):
    return DataLoader(
        CropMappingDataset(x, y, field_ids),
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

    def __init__(self, x, y, field_ids):
        self.x = x
        self.y = y
        self.field_ids  = field_ids 

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return {"sample_x": self.x[idx], "sample_y": self.y[idx], "sample_field_ids":self.field_ids[idx]}   

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

def getStamp(string_list):
    # Regular expression to extract the first 5-digit part of the ID
    pattern = re.compile(r"(\d{5})[_-]")

    # Extract locations
    ts_stamp_ls = []
    for filename in string_list:
        match = pattern.search(filename)  # Find the first part of the ID
        if match:
            first_part = match.group(1)  # Get the first 5-digit part
            second_part = filename[-9:-4]  # Extract the second part using slicing
            location = f"{first_part}_{second_part}"  # Combine both parts
            ts_stamp_ls.append(location)
    year_ls = [re.search(r'(20\d{2})', label_path).group(0) for label_path in string_list]
    
    return [year_ls[i]+'_'+ts_stamp_ls[i] for i in range(len(string_list))]


#%%

num_workers=2
drop_last=False
shuffle=False
_collate_fn=_collate_fn
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
input_feature_size,num_classes= 6,3
transformer = TransformerModel(seq_len = 23)
lr=0.0005
weight_decay=0.001
mode="min"
factor=0.1
patience=10
verbose=True
batch_size=int(pow(2, 12))
device = torch.device(f"cuda:{gpu_list[0]}" if torch.cuda.is_available() else "cpu")
classifier = transformer.to(device)
optimizer = torch.optim.Adam(classifier.parameters(), lr=0.0005)
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode=mode, factor=factor,
    patience=patience, verbose=verbose)
criterion = torch.nn.CrossEntropyLoss()
start_time = time.time()
end_time = time.time()
best_loss = float('inf')

def load_from_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def load_checkpoint(PATH):
    checkpoint = torch.load(PATH)
    classifier.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # epoch = checkpoint['epoch']
    # loss = checkpoint['loss']
    return classifier,optimizer
checkpoint_path = os.path.join(f'{best_models_directory1}/output_Anonymous_{fold_i}', f"{yy}_{preprocessing_method}_{model_type}_{sample_size}_best.pth")
try:
    classifier,optimizer = load_checkpoint(checkpoint_path)
    print(f'Using the best check point ...{checkpoint_path}')
except:
    pass

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
        
    batch_size=int(pow(2, 12))

    PATH = os.path.join(f'{best_models_directory1}/output_Anonymous_{fold_i}', f"{yy}_{preprocessing_method}_{sample_size}_scaler.pkl")
    scaler = load_from_pkl(PATH)
    x_test = normalize_with_scaler(scaler, x_test)
    y_test = np.zeros(x_test.shape[0])
    # test_loader = make_data_loader(x_test, y_test, shuffle=False,batch_size=batch_size,num_workers=num_workers,_collate_fn=_collate_fn,drop_last=False)# =test_dataloader
    field_ids_test = np.array(range(len(y_test)))
    test_loader = make_data_loader(x_test, y_test, field_ids_test, shuffle=False,batch_size=batch_size,num_workers=num_workers,_collate_fn=_collate_fn,drop_last=False)# =test_dataloader

    # prediction, _ = classifier(test_loader)
    start_time = time.time()
    end_time = time.time()
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
        end_time = time.time()
    

    predicting_time = end_time - start_time
    # Map predictions back to original image shape
    predicted_map = np.full(mask.shape, np.nan)
    predicted_map[mask] = prediction
    # print('predicted_map shape',predicted_map.shape)
    predicted_map_2d = predicted_map.reshape((height, width))
    # print('predicted_map_2d shape',predicted_map_2d.shape)

    print('predicting_time',predicting_time)
    # Save the output as a new GeoTiff with the same profile
    # output_file = "predicted_map.tif"
    output_file = os.path.join(f'./{preprocessing_method}_{model_type}', f"{ts_stamp}.tif")
    profile.update(dtype=rasterio.float32, count=1)

    with rasterio.open(output_file, "w", **profile) as dst:
        dst.write(predicted_map_2d.astype(rasterio.float32), 1)


createFolder(f'./{preprocessing_method}_{model_type}')
print(f'create Folder {preprocessing_method}_{model_type}')
# csv_path = '../csv_tif_list/28_29_'+preprocessing_method+'_tif_list.csv'
# file = pd.read_csv(csv_path)
# for i in range(file.shape[0]):
#     pred_map(file.iloc[i,:])

# pred_map('03500_02500','./7day_ln_image/2023-0000003500-0000002500.tif')
pred_map('02500_02000','./7day_ln_image/2023-0000002500-0000002000.tif')