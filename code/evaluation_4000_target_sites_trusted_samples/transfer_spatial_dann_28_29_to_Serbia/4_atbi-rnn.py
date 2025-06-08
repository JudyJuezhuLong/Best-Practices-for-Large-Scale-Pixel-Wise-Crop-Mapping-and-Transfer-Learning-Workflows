#%%
import matplotlib
print(matplotlib.__file__)
matplotlib.use('Agg')
import random
import time
import warnings
import argparse
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,precision_score,recall_score,cohen_kappa_score,f1_score
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD

from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

# Get the absolute path of the parent directory (assuming tllib is in the same level as experiment)
import sys
import json
import pandas as pd
import os
# Get the absolute path of the parent directory (assuming tllib is in the same level as experiment)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))
from examples.domain_adaptation.image_classification import utils
from tllib.modules.domain_discriminator import DomainDiscriminator
from tllib.alignment.dann import DomainAdversarialLoss
from tllib.utils.metric import accuracy
from tllib.utils.meter import AverageMeter, ProgressMeter
from tllib.utils.logger import CompleteLogger
# from models_DCM.classifier import Classifier
import numpy as np
from sklearn import preprocessing
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import re
import pickle

import matplotlib.pyplot as plt
experiment_name = "1_dann_"


# Automatically detect allocated GPUs
gpu_ids = os.environ.get("CUDA_VISIBLE_DEVICES", "0")  # Get assigned GPUs
gpu_list = [int(i) for i in gpu_ids.split(",")]

current_directory = os.path.dirname(os.path.abspath(__file__))
print(current_directory)
parent_l2_directory = os.path.dirname(current_directory)
parent_l1_directory = os.path.dirname(parent_l2_directory)

best_models_directory1 = parent_l1_directory + current_directory[len(parent_l2_directory):len(current_directory)]

yy='2023'
preprocessing_method='7days_ln'
model_type='atbi-rnn'
sample_size='4000'

source_scene_id = '28_29'
target_scene_id = 'Serbia'
scene_id = source_scene_id+'_'+target_scene_id

#%%
class RNN(nn.Module):
    def __init__(self, seed, input_feature_size, hidden_size, num_layers,
        bidirectional, dropout, num_classes,bottleneck_dim):
        super(RNN, self).__init__()
        self.modelname = self._get_name()
        self.hidden_size = hidden_size
        self.num_directions = bidirectional
        self.inlayernorm = nn.LayerNorm(input_feature_size)
        self.bottleneck_dim=bottleneck_dim


        self.rnn = nn.RNN(
            input_size=input_feature_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout,
        )
        num_directions = 2 if bidirectional else 1
        # self.decoder = nn.Linear(hidden_size * num_directions, num_classes, bias=True)
        self.decoder = nn.Linear(
            in_features=bottleneck_dim,
            out_features=num_classes,
        )
        self.attention = nn.Linear(
            in_features=num_directions * hidden_size,
            out_features=1,
        )
        self.test1=nn.Linear(num_directions * hidden_size, bottleneck_dim)
        self.test2=nn.BatchNorm1d(bottleneck_dim)
        self.relu=nn.ReLU()

    def get_parameters(self):
        """
        Return a parameters list which decides optimization hyper-parameters,
        such as the relative learning rate of each layer.
        """
        #params = [{"bottleneck": self.bottlenet, "lr": self.lr,"weight_decay":self.weight_decay}]
        params=[{"params":self.parameters()}]
        #params=[{name: param for name, param in self.named_parameters()}]
        return params
    
    def forward(self, x):
        # performs well when no this line
        # x = self.inlayernorm(x)
        
        self.rnn.flatten_parameters()
        rnn_out, _ = self.rnn(x)
        attn_weights = F.softmax(F.relu(self.attention(rnn_out)), dim=1)
        fc_in = attn_weights.permute(0, 2, 1).bmm(rnn_out)
        f1=self.test1(fc_in).squeeze()
        f2=self.test2(f1)
        f=self.relu(f2)        
        y_s = self.decoder(f)
        return y_s,f


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


def load_from_pth(path):
    return torch.load(path)


def drawAccuracyCurves(cls_accs_t_ls,cls_accs_ls,out_path):
    pd.DataFrame(cls_accs_t_ls).to_csv(out_path+'_cls_t_accs.csv', index=False)
    pd.DataFrame(cls_accs_ls).to_csv(out_path+'_cls_s_accs.csv', index=False)
    fig, ax = plt.subplots( nrows=1, ncols=1 )
    ax.plot(cls_accs_t_ls, label='Target domain training accuracy')
    ax.plot(cls_accs_ls, label='Source domain training accuracy')    
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
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


def main(args: argparse.Namespace):
    logger = CompleteLogger(args.log, args.phase)
    print(args)
    def build_model(input_feature_size,num_classes,seed=0,hidden_size=256, num_layers=2,bidirectional=True, dropout=0.2, bottleneck_dim=512):
        return RNN (seed=seed,
            input_feature_size=input_feature_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            num_classes=num_classes,
            bottleneck_dim = bottleneck_dim
            )
    def define_Fsize_and_Nclass(x,y):
        input_feature_size = x.shape[2]
        num_classes = np.unique(y).shape[0]
        return input_feature_size,num_classes

    def normalize_with_scaler(scaler, x_test):
        x_test = scaler(
            torch.FloatTensor(x_test.transpose((0, 2, 1)))
        ).numpy().transpose((0, 2, 1))
        return x_test
    
    def make_data_loader(x, y, shuffle,batch_size,num_workers,_collate_fn,drop_last):
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
    def _init_parameters(submodule):
        if type(submodule) == nn.RNN:
            for name, param in submodule.named_parameters():
                if re.search("bias_ih", name):
                    # set forget gate bias to 3.0
                    param.detach().chunk(4)[1].fill_(3.0)

    def init_parameters(net,):
        net.apply(_init_parameters)  


    with open("/mnt/mridata/Anonymouslong/best_practice_pixel/sample_data/input_path_config.json", "r") as f:
        file_paths = json.load(f)



    x_dir_key = 'x_data_dir_'+target_scene_id+'_'+preprocessing_method
    y_dir_key = 'y_data_dir_'+target_scene_id+'_'+preprocessing_method
    print("x_data path:",file_paths[x_dir_key])
    print("label path:",file_paths[y_dir_key])
    x_data = np.load(file_paths[x_dir_key])
    y_data = np.load(file_paths[y_dir_key]).ravel()
    print("X_data shape",x_data.shape)


    
    # Check for all NaN values in the second dimension
    any_nan_in_second_dim = np.any(np.isnan(x_data), axis=(1, 2))

    # Filter out the records where the second dimension contains all NaNs
    x_data = x_data[~any_nan_in_second_dim]
    y_data = y_data[~any_nan_in_second_dim]

    le = preprocessing.LabelEncoder()
    le.fit(y_data)
    y_data = le.transform(y_data)

    # Filter out NaNs using the boolean mask
    x_data[np.isnan(x_data)] = 0

    # Select 10,000 samples randomly from the dataset
    sample_size_target = 4000
    np.random.seed(42)
    random_indices = np.random.choice(len(x_data), size=sample_size_target, replace=False)

    x_test = x_data[random_indices]
    y_test = y_data[random_indices]
    print(f"Test size: {x_test.shape}")
   
    input_feature_size,num_classes=define_Fsize_and_Nclass(x_test,y_test)
    PATH =  os.path.join(f'{best_models_directory1}/output_Anonymous_{scene_id}site_dann', f"{yy}_{preprocessing_method}_{model_type}_{sample_size}_scaler.pkl")
    scaler = load_from_pkl(PATH)
    x_test = normalize_with_scaler(scaler, x_test)
   
    batch_size=32
    num_workers=2
    _collate_fn=_collate_fn
    device = torch.device(f"cuda:{gpu_list[0]}" if torch.cuda.is_available() else "cpu")
    print(f"Using GPU(s): {gpu_list}")
    
    test_loader = make_data_loader(x_test, y_test, shuffle=False,batch_size=batch_size,num_workers=num_workers,_collate_fn=_collate_fn,drop_last=False)
    classifier = build_model(input_feature_size,num_classes).to(device)
    init_parameters(classifier)
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    cudnn.benchmark = True


    domain_discri = DomainDiscriminator(in_feature=args.bottleneck_dim, hidden_size=1024).to(device)


    optimizer = SGD(classifier.get_parameters() + domain_discri.get_parameters(),
                    args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)



    def load_checkpoint(PATH):
        classifier.load_state_dict(torch.load(PATH))
        return classifier

    checkpoint_path = os.path.join(f'{best_models_directory1}/output_Anonymous_{scene_id}site_dann', f"{yy}_{preprocessing_method}_{model_type}_{sample_size}.pth")
    # print(checkpoint_path)
    try:
        classifier = load_checkpoint(checkpoint_path)
        print(f'Using the best check point ...{checkpoint_path}')
    except:
        pass

    yt_soft_pred_batch_list = []
    with torch.no_grad():
        for j, s in enumerate(test_loader):
            x_s = s["x"].to(device)
            optimizer.zero_grad()
            y_soft_pred_, f = classifier(x_s)
            yt_soft_pred_batch_list.append(F.softmax(y_soft_pred_, dim=1))
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
    os.makedirs(f'./output_Anonymous_dann', exist_ok=True)
    np.savetxt(os.path.join(f'./output_Anonymous_dann', f"{yy}_{preprocessing_method}_{model_type}_{sample_size}_confusion_matrix.csv"), confusion_matrix0, delimiter=",")

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
    metrics_df.to_csv(os.path.join(f'./output_Anonymous_dann', f"{yy}_{preprocessing_method}_{model_type}_{sample_size}_metrics.csv"), index=False)

    
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DANN for Unsupervised Domain Adaptation')
    # dataset parameters
    parser.add_argument('--root', metavar='DIR',default='/data/Anonymous/Anonymousfile/transfer_learning/Transfer-Learning-Library/data/office',
                        help='root path of dataset')
    parser.add_argument('-d', '--data', metavar='DATA', default='Office31', choices=utils.get_dataset_names(),
                        help='dataset: ' + ' | '.join(utils.get_dataset_names()) +
                             ' (default: Office31)')
    parser.add_argument('-s', '--source',default='A', help='source domain(s)', nargs='+')
    parser.add_argument('-t', '--target',default='W', help='target domain(s)', nargs='+')
    parser.add_argument('--train-resizing', type=str, default='default')
    parser.add_argument('--val-resizing', type=str, default='default')
    parser.add_argument('--resize-size', type=int, default=224,
                        help='the image size after resizing')
    parser.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                        help='Random resize scale (default: 0.08 1.0)')
    parser.add_argument('--ratio', type=float, nargs='+', default=[3. / 4., 4. / 3.], metavar='RATIO',
                        help='Random resize aspect ratio (default: 0.75 1.33)')
    parser.add_argument('--no-hflip', action='store_true',
                        help='no random horizontal flipping during training')
    parser.add_argument('--norm-mean', type=float, nargs='+',
                        default=(0.485, 0.456, 0.406), help='normalization mean')
    parser.add_argument('--norm-std', type=float, nargs='+',
                        default=(0.229, 0.224, 0.225), help='normalization std')
    # model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                        choices=utils.get_model_names(),
                        help='backbone architecture: ' +
                             ' | '.join(utils.get_model_names()) +
                             ' (default: resnet18)')
    parser.add_argument('--bottleneck-dim', default=512, type=int,
                        help='Dimension of bottleneck')
    parser.add_argument('--no-pool', action='store_true',
                        help='no pool layer after the feature extractor.')
    parser.add_argument('--scratch', action='store_true', help='whether train from scratch.')
    parser.add_argument('--trade-off', default=1., type=float,
                        help='the trade-off hyper-parameter for transfer loss')
    # training parameters
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N',
                        help='mini-batch size (default: 32)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lr-gamma', default=0.0005, type=float, help='parameter for lr scheduler')
    parser.add_argument('--lr-decay', default=0.75, type=float, help='parameter for lr scheduler')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-3, type=float,
                        metavar='W', help='weight decay (default: 1e-3)',
                        dest='weight_decay')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--epochs', default=10, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-i', '--iters-per-epoch', default=1000, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('-p', '--print-freq', default=1, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--per-class-eval', action='store_true',
                        help='whether output per-class accuracy during evaluation')
    parser.add_argument("--log", type=str, default='dann',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test', 'analysis'],
                        help="When phase is 'test', only test the model."
                             "When phase is 'analysis', only analysis the model.")
    args = parser.parse_args()
    main(args)