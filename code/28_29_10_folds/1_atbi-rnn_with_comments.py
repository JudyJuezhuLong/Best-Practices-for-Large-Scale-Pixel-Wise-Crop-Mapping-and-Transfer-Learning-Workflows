
import os
import sys
import re
import json
import time
import random
import pickle
import warnings
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, cohen_kappa_score, f1_score
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader

# Get the absolute path of the parent directory (assuming tllib is in the same level as experiment)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from examples.domain_adaptation.image_classification import utils
from tllib.utils.metric import accuracy
from tllib.utils.meter import AverageMeter, ProgressMeter

# Automatically detect allocated GPUs
gpu_ids = os.environ.get("CUDA_VISIBLE_DEVICES", "0")  # Get assigned GPUs
gpu_list = [int(i) for i in gpu_ids.split(",")]
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
yy='2023'
preprocessing_method='raw'
model_type='atbi-rnn'
sample_size='630k'
scene_id = '28_29'
#%%

class RNN(nn.Module):
    def __init__(self, seed, input_feature_size, hidden_size, num_layers,
        bidirectional, dropout, num_classes):
        super(RNN, self).__init__()
        self.modelname = self._get_name()
        self.hidden_size = hidden_size
        self.num_directions = bidirectional
        
        # Layer normalization for input features
        self.inlayernorm = nn.LayerNorm(input_feature_size)

        # RNN layer
        self.rnn = nn.RNN(
            input_size=input_feature_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout,
        )
        num_directions = 2 if bidirectional else 1

        # Linear layer for decoding the output of RNN
        self.decoder = nn.Linear(hidden_size * num_directions, num_classes, bias=True)

        # Attention mechanism
        self.attention = nn.Linear(
            in_features=num_directions * hidden_size,
            out_features=1,
        )

    def get_parameters(self):
        """
        Return a parameters list which decides optimization hyper-parameters,
        such as the relative learning rate of each layer.
        """
        params = [{"params": self.parameters()}]
        return params

    def forward(self, x):
        # Apply layer normalization to input
        x = self.inlayernorm(x)

        # Flatten parameters for RNN
        self.rnn.flatten_parameters()

        # Forward pass through RNN
        rnn_out, _ = self.rnn(x)
        
        # Compute attention weights
        attn_weights = F.softmax(F.relu(self.attention(rnn_out)), dim=1)

        # Apply attention weights to RNN output
        fc_in = attn_weights.permute(0, 2, 1).bmm(rnn_out)

        # Decode the output
        y_s = self.decoder(fc_in).squeeze()

        return y_s, rnn_out


def _assert_suffix_match(suffix, path):
    """
    Assert that the file path ends with the given suffix.
    """
    assert re.search(r"\.{}$".format(suffix), path), "suffix mismatch"

def make_parent_dir(filepath):
    """
    Create parent directories for the given file path if they do not exist.
    """
    parent_path = os.path.dirname(filepath)
    if not os.path.isdir(parent_path):
        try:
            os.mkdir(parent_path)
        except FileNotFoundError:
            make_parent_dir(parent_path)
            os.mkdir(parent_path)
        print("[INFO] Make new directory: '{}'".format(parent_path))

def save_to_pkl(data, path):
    """
    Save data to a pickle file.
    """
    _assert_suffix_match("pkl", path)
    make_parent_dir(path)
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=-1)
    print("[INFO] Save as pkl: '{}'".format(path))

def load_from_pkl(path):
    """
    Load data from a pickle file.
    """
    with open(path, "rb") as f:
        return pickle.load(f)

def save_to_pth(data, path, model=True):
    """
    Save data to a PyTorch file.
    """
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
    """
    Load data from a PyTorch file.
    """
    return torch.load(path)


def drawAccuracyCurves_(cls_accs_ls, out_path):
    """
    Draw and save accuracy curves.

    Args:
        cls_accs_ls (list): List of classification accuracies.
        out_path (str): Path to save the accuracy curve plot and CSV.
    """
    pd.DataFrame(cls_accs_ls).to_csv(out_path + '_cls_accs.csv', index=False)
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(cls_accs_ls, label='Source domain training accuracy')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0.7, 1)
    ax.legend()
    fig.savefig(out_path)
    plt.close(fig)


def drawLossCurves(loss_ls, out_path):
    """
    Draw and save loss curves.

    Args:
        loss_ls (list): List of training losses.
        out_path (str): Path to save the loss curve plot and CSV.
    """
    pd.DataFrame(loss_ls).to_csv(out_path + '_loss.csv', index=False)
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(loss_ls, label='Training loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    fig.savefig(out_path)
    plt.close(fig)


def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """
    Save model checkpoint.

    Args:
        model (nn.Module): The model to save.
        optimizer (Optimizer): The optimizer used for training.
        epoch (int): The current epoch number.
        loss (float): The loss value at the current epoch.
        filepath (str): The path to save the checkpoint.
    """
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

def main(args: argparse.Namespace):
    def build_model(input_feature_size, num_classes, seed=0, hidden_size=256, num_layers=2, bidirectional=True, dropout=0.2):
        """
        Build the RNN model with the specified parameters.

        Args:
            input_feature_size (int): Size of the input features.
            num_classes (int): Number of output classes.
            seed (int): Random seed for reproducibility.
            hidden_size (int): Number of features in the hidden state.
            num_layers (int): Number of recurrent layers.
            bidirectional (bool): If True, use bidirectional RNN.
            dropout (float): Dropout probability.

        Returns:
            RNN: An instance of the RNN model.
        """
        return RNN(
            seed=seed,
            input_feature_size=input_feature_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            num_classes=num_classes
        )
    
    def define_Fsize_and_Nclass(x, y):
        """
        Define the input feature size and number of classes.

        Args:
            x (np.ndarray): Input data.
            y (np.ndarray): Labels.

        Returns:
            tuple: (input_feature_size, num_classes)
        """
        input_feature_size = x.shape[2]
        num_classes = np.unique(y).shape[0]
        return input_feature_size, num_classes
    
    def normalize_without_scaler(x_train, x_test):
        """
        Normalize the training and test data without using a scaler.

        Args:
            x_train (np.ndarray): Training data.
            x_test (np.ndarray): Test data.

        Returns:
            tuple: (scaler, normalized_x_train, normalized_x_test)
        """
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
        """
        Normalize the test data using the provided scaler.

        Args:
            scaler (torch.nn.Module): Scaler used for normalization.
            x_test (np.ndarray): Test data.

        Returns:
            np.ndarray: Normalized test data.
        """
        x_test = scaler(
            torch.FloatTensor(x_test.transpose((0, 2, 1)))
        ).numpy().transpose((0, 2, 1))
        return x_test

    def make_data_loader(x, y, shuffle, batch_size, num_workers, _collate_fn, drop_last):
        """
        Create a DataLoader for the dataset.

        Args:
            x (np.ndarray): Input data.
            y (np.ndarray): Labels.
            shuffle (bool): Whether to shuffle the data.
            batch_size (int): Number of samples per batch.
            num_workers (int): Number of subprocesses to use for data loading.
            _collate_fn (callable): Function to merge a list of samples to form a mini-batch.
            drop_last (bool): Whether to drop the last incomplete batch.

        Returns:
            DataLoader: DataLoader for the dataset.
        """
        return DataLoader(
            CropMappingDataset(x, y),
            batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, collate_fn=_collate_fn,
            drop_last=drop_last
        )
    
    def _collate_fn(batch):
        """
        Define how to aggregate samples into a batch.

        Args:
            batch (list): List of samples.

        Returns:
            dict: Dictionary containing batched input data and labels.
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
        Crop classification dataset.
        """
    
        def __init__(self, x, y):
            """
            Initialize the dataset with input data and labels.

            Args:
                x (np.ndarray): Input data.
                y (np.ndarray): Labels.
            """
            self.x = x
            self.y = y
    
        def __len__(self):
            """
            Return the number of samples in the dataset.

            Returns:
                int: Number of samples.
            """
            return self.x.shape[0]
    
        def __getitem__(self, idx):
            """
            Get a sample from the dataset at the specified index.

            Args:
                idx (int): Index of the sample.

            Returns:
                dict: Dictionary containing input data and label.
            """
            return {"sample_x": self.x[idx], "sample_y": self.y[idx]}   
         
    def _init_parameters(submodule):
        """
        Initialize parameters for LSTM submodules.

        Args:
            submodule (nn.Module): Submodule to initialize.
        """
        if type(submodule) == nn.LSTM:
            for name, param in submodule.named_parameters():
                if re.search("bias_ih", name):
                    # Set forget gate bias to 3.0
                    param.detach().chunk(4)[1].fill_(3.0)
    
    def init_parameters(net):
        """
        Initialize parameters for the entire network.

        Args:
            net (nn.Module): Network to initialize.
        """
        net.apply(_init_parameters) 

    def _eval_perf(net, dataloader, best_loss, device):
        """
        Evaluate the performance of the model on the validation set.

        Args:
            net (nn.Module): The model to evaluate.
            dataloader (DataLoader): DataLoader for the validation set.
            best_loss (float): The best validation loss observed so far.
            device (torch.device): The device to run the evaluation on.

        Returns:
            tuple: (val_loss, val_correct, best_loss)
        """
        # Set the model to evaluation mode
        net.eval()
        val_loss = 0
        val_correct = 0

        # Disable gradient computation
        with torch.no_grad():
            for j, s in enumerate(dataloader):
                x_s = s["x"].to(device)
                labels_s = s["y"].to(device)
                y_s, f = net(x_s)
                val_loss += criterion(y_s, labels_s).item()
                yt_pred = torch.max(y_s, dim=1)[1]
                val_correct += (yt_pred == labels_s).sum().item()

        # Compute average validation loss and accuracy
        val_loss /= len(val_loader)
        val_correct /= len(val_loader)

        # Save checkpoint if validation loss improves
        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(net, optimizer, epoch, val_loss, os.path.join(f'./output_Anonymous_{fold_i}', f"{yy}_{preprocessing_method}_{model_type}_{sample_size}_best.pth"))

        return val_loss, val_correct, best_loss

    # Load file paths from configuration file
    with open("input_path_config.json", "r") as f:
        file_paths = json.load(f)

    # Define keys for x and y data paths
    x_dir_key = 'x_data_dir_'+scene_id+'_'+preprocessing_method
    y_dir_key = 'y_data_dir_'+scene_id+'_'+preprocessing_method
    print("X_data path:", file_paths[x_dir_key])
    print("label path:", file_paths[y_dir_key])

    # Load x and y data
    x_data = np.load(file_paths[x_dir_key])
    y_data = np.load(file_paths[y_dir_key]).ravel()
    print("X_data shape", x_data.shape)

    # Check for NaN values in the second dimension
    any_nan_in_second_dim = np.any(np.isnan(x_data), axis=(1, 2))

    # Filter out records with NaN values in the second dimension
    x_data = x_data[~any_nan_in_second_dim]
    y_data = y_data[~any_nan_in_second_dim]
    print(x_data.shape)

    # Encode labels
    le = preprocessing.LabelEncoder()
    le.fit(y_data)
    y_data = le.transform(y_data)

    # Replace NaN values with 0
    x_data[np.isnan(x_data)] = 0

    # Select a random sample of 400,000 samples from the dataset
    sample_size = 400000
    random_indices = np.random.choice(len(x_data), size=sample_size, replace=False)
    x_data = x_data[random_indices]
    y_data = y_data[random_indices]

    # Number of folds for cross-validation
    n_folds = 10

    # Initialize KFold cross-validator
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=18)
    folds = list(kf.split(x_data))

    # Iterate over each fold
    for fold_i in range(n_folds):
        # Define test set
        test_idx = folds[fold_i][1]

        # Define remaining folds for training and validation
        remaining_folds = [folds[j][1] for j in range(n_folds) if j != fold_i]
        val_idx = remaining_folds[0]
        train_idx = np.concatenate(remaining_folds[1:])

        # Split data into train, validation, and test sets
        x_train, y_train = x_data[train_idx], y_data[train_idx]
        x_val, y_val = x_data[val_idx], y_data[val_idx]
        x_test, y_test = x_data[test_idx], y_data[test_idx]

        # Combine training and validation data for scaling
        x_train_val = np.concatenate((x_train, x_val))
        y_train_val = np.concatenate((y_train, y_val))

        # Print sizes for each split
        print(f"Experiment {fold_i+1}:")
        print(f"Train size: {x_train.shape}, Validation size: {x_val.shape}, Test size: {x_test.shape}")

        # Define batch size
        batch_size = int(pow(2, 12))

        # Print training data size
        print("x_train size", x_train.shape)

        # Define input feature size and number of classes
        input_feature_size, num_classes = define_Fsize_and_Nclass(x_train_val, y_train_val)

        # Normalize data
        scaler, x_train_val, x_test = normalize_without_scaler(x_train_val, x_test)
        x_train = normalize_with_scaler(scaler, x_train)
        x_val = normalize_with_scaler(scaler, x_val)

        # Define DataLoader parameters
        num_workers = 2
        _collate_fn = _collate_fn
        device = torch.device(f"cuda:{gpu_list[0]}" if torch.cuda.is_available() else "cpu")
        print(f"Using GPU(s): {gpu_list}")

        # Create DataLoaders for training, validation, and test sets
        train_loader = make_data_loader(x_train, y_train, shuffle=True, batch_size=batch_size, num_workers=num_workers, _collate_fn=_collate_fn, drop_last=True)
        val_loader = make_data_loader(x_val, y_val, shuffle=True, batch_size=batch_size, num_workers=num_workers, _collate_fn=_collate_fn, drop_last=True)
        test_loader = make_data_loader(x_test, y_test, shuffle=False, batch_size=batch_size, num_workers=num_workers, _collate_fn=_collate_fn, drop_last=False)

        # Build and initialize the model
        classifier = build_model(input_feature_size, num_classes).to(device)
        init_parameters(classifier)

        # Set random seed for reproducibility
        if args.seed is not None:
            random.seed(args.seed)
            torch.manual_seed(args.seed)
            cudnn.deterministic = True
            warnings.warn('You have chosen to seed training. This will turn on the CUDNN deterministic setting, which can slow down your training considerably! You may see unexpected behavior when restarting from checkpoints.')

        cudnn.benchmark = True

        # Define optimizer and learning rate scheduler
        lr = 0.0005
        weight_decay = 0.001
        mode = "min"
        factor = 0.1
        patience = 10
        verbose = True

        optimizer = optim.Adam(classifier.get_parameters(), lr=lr, weight_decay=weight_decay)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience, verbose=verbose)

        # Define loss function
        criterion = nn.CrossEntropyLoss()

        # Initialize lists to store accuracy and loss values
        cls_accs_ls = []
        loss_ls = []

        # Record start time
        start_time = time.time()
        end_time = time.time()
        best_loss = float('inf')

        # Function to load checkpoint
        def load_checkpoint(PATH):
            checkpoint = torch.load(PATH)
            classifier.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            return classifier, optimizer

        # Load the best checkpoint if available
        checkpoint_path = os.path.join(f'./output_Anonymous_{fold_i}', f"{yy}_{preprocessing_method}_{model_type}_{sample_size}_best.pth")
        try:
            classifier, optimizer = load_checkpoint(checkpoint_path)
            print('Using the best check point ...')
        except:
            pass

        # Training loop
        for epoch in range(args.epochs):
            classifier.train()
            batch_time = AverageMeter('Time', ':5.2f')
            data_time = AverageMeter('Data', ':5.2f')
            losses = AverageMeter('Loss', ':6.2f')
            cls_accs = AverageMeter('Cls Acc', ':3.1f')
            progress = ProgressMeter(len(train_loader), [batch_time, data_time, losses, cls_accs], prefix="Epoch: [{}]".format(epoch))

            end_time = time.time()

            # Iterate over training batches
            for j, s in enumerate(train_loader):
                x_s = s["x"].to(device)
                labels_s = s["y"].to(device)
                optimizer.zero_grad()
                y_s, f = classifier(x_s)
                loss = criterion(y_s, labels_s)
                cls_acc = accuracy(y_s, labels_s)[0]

                losses.update(loss.item(), x_s.size(0))
                cls_accs.update(cls_acc.item(), x_s.size(0))

                loss.backward()
                optimizer.step()

                # Measure elapsed time
                batch_time.update(time.time() - end_time)
                end_time = time.time()

            # Update learning rate scheduler
            lr_scheduler.step(loss.item())

            # Evaluate model performance on validation set
            val_loss, val_correct, best_loss = _eval_perf(classifier, val_loader, best_loss, device)
            cls_accs_ls.append(val_correct)
            loss_ls.append(val_loss)

        # Record training time
        training_time = end_time - start_time
        print("training_time", training_time)

        # Save the trained model
        save_to_pth(classifier, os.path.join(f'./output_Anonymous_{fold_i}', f"{yy}_{preprocessing_method}_{model_type}_{sample_size}.pth"))

        # Draw accuracy and loss curves
        drawAccuracyCurves_(cls_accs_ls, os.path.join(f'./output_Anonymous_{fold_i}', f"{yy}_{preprocessing_method}_{model_type}_{sample_size}_acc.png"))
        drawLossCurves(loss_ls, os.path.join(f'./output_Anonymous_{fold_i}', f"{yy}_{preprocessing_method}_{model_type}_{sample_size}_loss.png"))

        # Evaluate model on test set
        yt_soft_pred_batch_list = []
        classifier, optimizer = load_checkpoint(checkpoint_path)
        with torch.no_grad():
            for j, s in enumerate(test_loader):
                x_s = s["x"].to(device)
                labels_s = s["y"].to(device)
                optimizer.zero_grad()
                y_soft_pred_, f = classifier(x_s)
                yt_soft_pred_batch_list.append(F.softmax(y_soft_pred_, dim=1))
            y_soft_pred = torch.cat(yt_soft_pred_batch_list, dim=0).cpu().numpy()
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


#%%
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
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-i', '--iters-per-epoch', default=1000, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('-p', '--print-freq', default=1000, type=int,
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