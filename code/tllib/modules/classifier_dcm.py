import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DCM(nn.Module):
    def __init__(
        self, seed, input_feature_size, hidden_size, num_layers,
        bidirectional, dropout, num_classes, bottleneck_dim,
    ):
        super().__init__()
        self._set_reproducible(seed)

        self.lstm = nn.LSTM(
            input_size=input_feature_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout,
        )  # i/o: (batch, seq_len, num_directions*input_/hidden_size)
        num_directions = 2 if bidirectional else 1
        self.attention = nn.Linear(
            in_features=num_directions * hidden_size,
            out_features=1,
        )
        self.bottleneck_dim=bottleneck_dim
        self.bottlenet=nn.Sequential(
            # nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            # nn.Flatten(),
            nn.Linear(num_directions * hidden_size, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU()
        )
        self.head = nn.Linear(
            in_features=bottleneck_dim,
            out_features =num_classes,
        )
        self.test_1=nn.Linear(num_directions * hidden_size, bottleneck_dim)
        self.test_2=nn.BatchNorm1d(bottleneck_dim)
    def _set_reproducible(self, seed, cudnn=False):
        np.random.seed(seed)
        torch.manual_seed(seed)
        if cudnn:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def forward(self, x):
        self.lstm.flatten_parameters()
        # lstm_out: (batch, seq_len, num_directions*hidden_size)
        lstm_out, _ = self.lstm(x)
        # softmax along seq_len axis
        attn_weights = F.softmax(F.relu(self.attention(lstm_out)), dim=1)
        # attn (after permutation): (batch, 1, seq_len)
        fc_in = attn_weights.permute(0, 2, 1).bmm(lstm_out)
        test_1=self.test_1(fc_in)
        test_2=self.test_2(test_1)
        f = self.bottlenet(fc_in)
        predictions = self.head(f)
        return predictions, f