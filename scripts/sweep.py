import subprocess
import numpy as np
import pandas as pd
from pandas import DataFrame
from tqdm import tqdm
from collections import defaultdict
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "2"    # GPU 2 - Randall
sys.path.append(r'/home/rlfowler/Documents/research/TFO-Inverse-Modeling')
from mtools import MLP
import torch.nn as nn
import torch
import itertools
from sklearn import preprocessing
from mtools import RandomSplit, custom_holdout, get_individual_criterion
from mtools import set_seed, data_filter1, total_counter, DataLoaderGenerator
from mtools import get_loss_fig
from mtools.display import get_error_distrubition_fig, get_error_stats

# Iterator skips for the sweep
SKIP_TO = 0
SKIP_LIST = []


# List of data paramters to sweep over
data_params = defaultdict(list)
data_params['output_labels'] = [[*range(7)]] # Column indices of the output labels   (none for all columns)
data_params['input_labels'] = [None]         # Column indices of the input labels    (none for all columns)  
data_params['random_seed'] = [42]            # Random seed for the model (not implemented currently)
data_params['log_transform'] = [True]        # Log transform the data
data_params['batch_size'] = [32, 512]           # Batch size for training
data_params['filter_method'] = [data_filter1]       # Method for filtering data
data_params['validation_method'] = [custom_holdout(0), RandomSplit(0.8)]    # Method for splitting data into training and validation sets


# List of training parameters to sweep over
train_params = defaultdict(list)
train_params['num_epochs'] = [3]             # Number of epochs for training
train_params['learning_rate'] = [5e-4]        # Learning rate for the model
train_params['weight_decay'] = [0]            # Weight decay for the optimizer
train_params['optimizer'] = [torch.optim.SGD] # Optimizer for the model
train_params['loss_func'] = [nn.MSELoss]      # Loss function for the model
train_params['loss_tracker'] = [get_individual_criterion]         # Loss tracker for the model


# List of model parameters to sweep over
model_params = defaultdict(list)
model_params['model'] = [MLP]       # Model class to use
model_params['hidden_layers'] = [   # Hidden layer sizes for the linear layers (not including input and output layers)
    #[40, 30, 20, 10],
    [30, 10]
    ]
model_params['activation'] = [      # Activation function for the hidden layers
    [nn.ReLU()],
    ]
model_params['dropout'] = [         # Dropout rate for the hidden layers
    [0],
    ]
model_params['batch_norm'] = [True] # Batch normalization for the hidden layers


# Plotting parameters (not for sweeping)
plot_params = {}
plot_params['loss_log'] = False     # Logarithmic scale for the loss plots
plot_params['loss_title'] = None    # Title for the loss plot
plot_params['loss_xlabel'] = "Epoch"# X-axis label for the loss plot
plot_params['loss_ylabel'] = "Loss" # Y-axis label for the loss plot
plot_params['loss_legend'] = True   # Include a legend in the loss plot

plot_params['error_distribution'] = True    # Plot the error distribution
plot_params['error_resolution'] = 4096      # Resolution of the error distribution plots (batch size)
plot_params['plot_bins'] = 10               # Number of bins for the error distribution plots


# Constants in Sweep
DATA_PATH = r'/home/rlfowler/Documents/research/tfo_inverse_modelling/Randalls Folder/data/randall_data_intensities.pkl'
COPY_PICKLE = True            # Create copy of data for each filtering (True uses more memory, False uses more time)
LABEL_NAMES = ['Maternal Wall Thickness', 'Fetal Radius', 'Fetal Displacement', 'Maternal Hb Concentration', 'Maternal Saturation', 'Fetal Hb Concentration', 'Fetal Saturation']
LABEL_START_INDEX = len(LABEL_NAMES)       # All columns before this index are considered output features (7)
DATA_LOADER_PARAMS = None   # Default set if none

JUMPS = [total_counter(train_params, model_params) * total_counter(data_params) / (len(data_params['filter_method']) * len(data_params['log_transform'])),
         total_counter(train_params, model_params)]


if __name__ == "__main__":
    iter = 0
    with tqdm(total=total_counter(data_params, train_params, model_params), desc="Sweeping") as pbar:
        if COPY_PICKLE:
            df:DataFrame = pd.read_pickle(DATA_PATH)
        for filter_method, apply_log in itertools.product(data_params['filter_method'], data_params['log_transform']):
            if iter + JUMPS[0] < SKIP_TO:
                iter += JUMPS[0]
                pbar.update(JUMPS[0])
                continue

            try:
                # Read data and filter
                if COPY_PICKLE:
                    data = df.copy()
                else:
                    data:DataFrame = pd.read_pickle(DATA_PATH)
                data = filter_method(data)

                # Get input and output columns
                x_columns = data.columns[LABEL_START_INDEX:]    # Input columns
                y_columns = data.columns[:LABEL_START_INDEX]    # Output columns

                # Apply log transformation to the data
                if apply_log:
                    data[x_columns] = np.log(data[x_columns])

                # Normalize the data
                y_scalar = preprocessing.StandardScaler()
                data[y_columns] = y_scalar.fit_transform(data[y_columns])
                x_scalar = preprocessing.StandardScaler()
                data[x_columns] = x_scalar.fit_transform(data[x_columns])

                for data_params_tuple in itertools.product(*[data_params[key] for key in data_params.keys() if key != 'filter_method' and key != 'log_transform']):
                    if iter + JUMPS[1] < SKIP_TO:
                        iter += JUMPS[1]
                        pbar.update(JUMPS[1])
                        continue
                    
                    try:
                        output_labels, input_labels, random_seed, batch_size, validation_method = data_params_tuple
                        # Fix label indices
                        if output_labels is None:
                            output_labels = [*range(len(y_columns))]
                        if input_labels is None:
                            input_labels = [*range(len(x_columns))]

                        # Set the seed
                        #set_seed(random_seed)

                        # Create data loaders
                        DLG = DataLoaderGenerator(data, x_columns[input_labels], y_columns[output_labels], validation_method, batch_size, DATA_LOADER_PARAMS)
                        train_loader, val_loader = DLG.generate()

                        # Prepare data for error distribution plots
                        if plot_params['error_distribution']:
                            train_loader2, val_loader2 = DLG.generate(batch_size=plot_params['error_resolution'])
                            train_data = y_scalar.inverse_transform(train_loader.dataset[:][1].cpu())
                            val_data = y_scalar.inverse_transform(val_loader.dataset[:][1].cpu())

                        for model_training_params_tuple in itertools.product(*[model_params[key] for key in model_params.keys()], *[train_params[key] for key in train_params.keys()]): # Maybe remove num_epochs and have ongoing training?
                            if iter < SKIP_TO or iter in SKIP_LIST:
                                iter += 1
                                pbar.update()
                                continue
                            
                            try:
                                model_class, hidden_layers, activation, dropout, batch_norm, num_epochs, learning_rate, weight_decay, optimizer_class, loss_func, loss_tracker = model_training_params_tuple
                                
                                # Get loss function tracker
                                if loss_func is not None:
                                    loss_func = loss_tracker(loss_func, y_columns, output_labels)   # Change name and params?

                                # Create Model
                                model = model_class(
                                    node_counts=[len(x_columns[input_labels])] + hidden_layers + [len(y_columns[output_labels])],
                                    dropout_rates=dropout,
                                    batch_norm=batch_norm,
                                    act_funcs=activation,
                                    validation_method=validation_method,
                                    loss_func=loss_func,
                                )

                                # Get optimizer
                                # if 'betas' in optimizer_class.__init__.__code__.co_varnames: # Use this if optimizer has betas
                                optimizer = optimizer_class(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
                                
                                # Train the model
                                model.run_training(optimizer, train_loader, val_loader, num_epochs)

                                # Plot results
                                loss_fig = get_loss_fig(loss_func, log=plot_params['loss_log'], title=plot_params['loss_title'], xlabel=plot_params['loss_xlabel'], ylabel=plot_params['loss_ylabel'], legend=plot_params['loss_legend'])
                                if plot_params['error_distribution']:
                                    error_fig, train_stats, val_stats = get_error_distrubition_fig(model, train_loader2, val_loader2, y_columns, y_scalar, train_data, val_data, plot_params['plot_bins'])
                                else:
                                    error_fig = None
                                    train_stats, val_stats = get_error_stats(model, train_loader, val_loader, y_columns, y_scalar)
                                dist_metrics_fig = None

                                iter += 1
                                pbar.update()

                            except Exception as e:
                                print(f"Error preparing model: {e}")
                                pbar.update()
                                continue

                    except Exception as e:
                        print(f"Error preparing data: {e}")
                        iter += JUMPS[1]
                        pbar.update(JUMPS[1])
                        continue

            except Exception as e:
                print(f"Error filtering data: {e}")
                iter += JUMPS[0]
                pbar.update(JUMPS[0])
                continue

