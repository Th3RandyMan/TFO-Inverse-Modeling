import subprocess
from tqdm import tqdm
from collections import defaultdict
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
sys.path.append(r'/home/rlfowler/Documents/research/tfo_inverse_modelling')
from ..models.models import MLP
import torch.nn as nn
import torch
from ..models.utils import set_seed

# List of data paramters to sweep over
data_params = defaultdict(list)
data_params['output_labels'] = [[*range(7)]] # Column indices of the output labels   (none for all columns)
data_params['input_labels'] = [None]         # Column indices of the input labels    (none for all columns)  
data_params['random_seed'] = [42]             # Random seed for the model
data_params['batch_size'] = [32, 512]                  # Batch size for training

# List of training parameters to sweep over
train_params = defaultdict(list)
train_params['num_epochs'] = [25]                  # Number of epochs for training
train_params['learning_rate'] = [5e-4]        # Learning rate for the model
train_params['weight_decay'] = [0]            # Weight decay for the optimizer

# List of model parameters to sweep over
model_params = defaultdict(list)
model_params['model'] = [MLP]
model_params['hidden_layers'] = [   # Hidden layer sizes for the linear layers (not including input and output layers)
    [40, 30, 20, 10],
    ]
model_params['activation'] = [   # Activation function for the hidden layers
    [nn.ReLU()],
    ]
model_params['dropout'] = [   # Dropout rate for the hidden layers
    [0],
    ]
model_params['batch_norm'] = [True]   # Batch normalization for the hidden layers

# Constants in Sweep
LABEL_START_INDEX = 7   # All columns before this index are considered output features
DATA_LOADER_PARAMS = None # Default set if none



if __name__ == "__main__":
    pass
