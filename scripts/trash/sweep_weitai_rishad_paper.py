"""
Author: Randall Fowler
Date: 2024-08-28
Description: Script to sweep over data, training, and model parameters for inverse modeling

To Do:
    - Print flush for stdout
    - In report, have legend for changes in iteration
    - Ticks on plots for integer values
    - Use pulsation data rather than intensity
"""
from typing import Callable
import numpy as np
import pandas as pd
from pandas import DataFrame
from tqdm import tqdm
from collections import defaultdict
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "2"    # GPU 2 - Randall
sys.path.append(r'/home/rlfowler/Documents/research/TFO-Inverse-Modeling')

from DLTools.modeling import BaseModel, MLP
import torch.nn as nn
import torch
import itertools
from sklearn import preprocessing
from DLTools.processing import DataLoaderGenerator
from DLTools.processing import RandomSplit, custom_holdout, get_individual_criterion
from custom.filters import data_filter_remove_fr, data_filter_remove_fd, CombineFilters
from custom.display import dict_str, get_error_distrubition_fig, get_error_stats, get_loss_row, get_stats_row, plot_stats, get_loss_fig
from custom.misc import total_counter
# from mdreport import MarkdownReport
from PyMD import MDGenerator
from pathlib import Path


# RUN PARAMETERS
# RUN_NAME = "wr_paper_short_epr1"
# RUN_NAME = "wr_paper_large_epr1"
RUN_NAME = "wr_paper_weitai_epr1"
TITLE = RUN_NAME + " - Inverse Modeling"
BASE_NAME = "reports/"
LOSS_FOLDER = "losses/"
DIST_FOLDER = "distributions/"
SAVED_MODELS_FOLDER = "saved_models/"
SKIP_TO = 0
SKIP_LIST = []


# List of data paramters to sweep over
data_params = defaultdict(list)
#data_params['output_labels'] = [7] # Column indices of the output labels   (none for all columns)
data_params['output_labels'] = [3]  # For weitai data
data_params['input_labels'] = [[1, 5, 8, 13, 19, 21, 25, 28, 33, 39]]         # Column indices of the input labels    (none for all columns)  
data_params['random_seed'] = [42]            # Random seed for the model (not implemented currently)
data_params['log_transform'] = [False]        # Log transform the data
data_params['batch_size'] = [64]           # Batch size for training
# data_params['filter_method'] = [CombineFilters(data_filter_remove_fr, data_filter_remove_fd)]       # Method for filtering data
data_params['filter_method'] = [None]
data_params['validation_method'] = [RandomSplit(0.8)]    # Method for splitting data into training and validation sets


# List of training parameters to sweep over
train_params = defaultdict(list)
train_params['num_epochs'] = [300]             # Number of epochs for training
train_params['learning_rate'] = [1e-4]        # Learning rate for the model
train_params['weight_decay'] = [1e-4]            # Weight decay for the optimizer
train_params['optimizer'] = [torch.optim.Adam] # Optimizer for the model
train_params['loss_func'] = [nn.MSELoss]      # Loss function for the model
train_params['loss_tracker'] = [get_individual_criterion]         # Loss tracker for the model


# List of model parameters to sweep over
model_params = defaultdict(list)
model_params['model'] = [MLP]       # Model class to use
model_params['hidden_layers'] = [   # Hidden layer sizes for the linear layers (not including input and output layers)
    [64, 32, 16, 8],
    ]
model_params['activation'] = [      # Activation function for the hidden layers
    [nn.ReLU()],
    ]
model_params['dropout'] = [         # Dropout rate for the hidden layers
    False,
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
# DATA_PATH = r'/home/rlfowler/Documents/research/TFO-Simulator/data/randall_data_EPR.parquet'
# DATA_PATH = r'/home/rlfowler/Documents/research/TFO-Simulator/data/randall_short_EPR.parquet'
DATA_PATH = r'/home/rlfowler/Documents/research/TFO-Simulator/data/weitai_epr.parquet'
COPY_DATA = True            # Create copy of data for each filtering (True uses more memory, False uses more time)
# LABEL_NAMES = ['Maternal Wall Thickness', 'Fetal Radius', 'Fetal Displacement', 'Maternal Hb Concentration', 'Maternal Saturation', 'Fetal Hb Concentration 1', 'Fetal Hb Concentration 2', 'Fetal Saturation']
LABEL_NAMES = ['Maternal Wall Thickness', 'Maternal Hb Concentration', 'Maternal Saturation', 'Fetal Saturation', 'Fetal Hb Concentration 1', 'Fetal Hb Concentration 2']
LABEL_START_INDEX = len(LABEL_NAMES)       # All columns before this index are considered output features (7)
DATA_LOADER_PARAMS = None   # Default set if none

JUMPS = [total_counter(train_params, model_params) * total_counter(data_params) / (len(data_params['filter_method']) * len(data_params['log_transform'])),
         total_counter(train_params, model_params)]
STATS_COLUMNS = [label + ' ' + stat for label in LABEL_NAMES + [''] for stat in ['Train Mean', 'Train Std', 'Val Mean', 'Val Std']]
LOSS_COLUMNS = [label + ' ' + stat for label in LABEL_NAMES + [''] for stat in ['Train Loss', 'Val Loss']]

if __name__ == "__main__":
    read_data:Callable = pd.read_pickle if DATA_PATH.endswith('.pkl') else pd.read_parquet

    total = total_counter(data_params, train_params, model_params)
    stats_df = pd.DataFrame(index=range(total), columns=STATS_COLUMNS)
    loss_df = pd.DataFrame(index=range(total), columns=LOSS_COLUMNS)
    Path(BASE_NAME+LOSS_FOLDER).mkdir(exist_ok=True)
    Path(BASE_NAME+DIST_FOLDER).mkdir(exist_ok=True)
    md = MDGenerator(Path(BASE_NAME), RUN_NAME, TITLE)
    md["Sweep Parameters"].add_code(f"Data Parameters: \n\t{dict_str(data_params)}\n\nTraining Parameters: \n\t{dict_str(train_params)}\n\nModel Parameters: \n\t{dict_str(model_params)}")
    md["Plotting Parameters"].add_code(f"Plot Parameters: \n\t{dict_str(plot_params)}")
    # report = MarkdownReport(Path(BASE_NAME), RUN_NAME, TITLE)
    # report.add_code_report("Sweep Parameters", f"Data Parameters: \n\t{dict_str(data_params)}\n\nTraining Parameters: \n\t{dict_str(train_params)}\n\nModel Parameters: \n\t{dict_str(model_params)}")
    # report.add_code_report("Plotting Parameters", f"Plot Parameters: \n\t{dict_str(plot_params)}")

    iter = 0
    best_loss = float('inf')
    best_model = None
    with tqdm(total=total, desc="Sweeping") as pbar:
        if COPY_DATA:
            df:DataFrame = read_data(DATA_PATH)
        for filter_method, apply_log in itertools.product(data_params['filter_method'], data_params['log_transform']):
            if iter + JUMPS[0] < SKIP_TO:
                iter += JUMPS[0]
                pbar.update(JUMPS[0])
                continue

            try:
                # Read data and filter
                if COPY_DATA:
                    data = df.copy()
                else:
                    data:DataFrame = read_data(DATA_PATH)

                # Filter the data
                if filter_method is not None:
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
                        elif isinstance(output_labels, int):
                            output_labels = [output_labels]
                        else:
                            pass # add method for string labels

                        if input_labels is None:
                            input_labels = [*range(len(x_columns))]
                        elif isinstance(input_labels, int):
                            input_labels = [input_labels]
                        else:
                            pass # add method for string labels

                        # Set the seed
                        #set_seed(random_seed)

                        # Create data loaders
                        DLG = DataLoaderGenerator(data, x_columns[input_labels], y_columns[output_labels], validation_method, batch_size, DATA_LOADER_PARAMS)
                        train_loader, val_loader = DLG.generate()

                        # Prepare data for error distribution plots
                        if plot_params['error_distribution']:
                            train_loader2, val_loader2 = DLG.generate(batch_size=plot_params['error_resolution'])
                            # train_data = y_scalar.inverse_transform(train_loader.dataset[:][1].cpu())
                            # val_data = y_scalar.inverse_transform(val_loader.dataset[:][1].cpu())
                            # Replaced with below to use inverse_transform, but needed to include multiple labels since only one is used. Not ideal, but works.
                            train_data = y_scalar.inverse_transform(torch.cat([train_loader.dataset[:][1].cpu()]*len(y_scalar.feature_names_in_), dim=1))[:,output_labels]
                            val_data = y_scalar.inverse_transform(torch.cat([val_loader.dataset[:][1].cpu()]*len(y_scalar.feature_names_in_), dim=1))[:,output_labels]

                        for training_model_params_tuple in itertools.product(*[train_params[key] for key in train_params.keys()], *[model_params[key] for key in model_params.keys()]): # Maybe remove num_epochs and have ongoing training?
                            if iter < SKIP_TO or iter in SKIP_LIST:
                                iter += 1
                                pbar.update()
                                continue
                            
                            try:
                                num_epochs, learning_rate, weight_decay, optimizer_class, loss_func, loss_tracker, model_class, hidden_layers, activation, dropout, batch_norm = training_model_params_tuple
                                print(f"\n*** Run {iter+1}/{total} ***", flush=True)
                                
                                # Get loss function tracker
                                if loss_func is not None:
                                    loss_func = loss_tracker(loss_func, y_columns[output_labels])   # Track all of them for now...

                                # Create Model
                                model:BaseModel = model_class(
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
                                val_loss = model.run_training(optimizer, train_loader, val_loader, num_epochs, verbose=True)
                                if best_loss > val_loss:
                                    best_loss = val_loss
                                    best_model = model

                                # Plot results
                                loss_fig = get_loss_fig(loss_func, log=plot_params['loss_log'], title=plot_params['loss_title'], xlabel=plot_params['loss_xlabel'], ylabel=plot_params['loss_ylabel'], legend=plot_params['loss_legend'])
                                if plot_params['error_distribution']:
                                    error_fig, train_stats, val_stats = get_error_distrubition_fig(model, train_loader2, val_loader2, y_columns[output_labels], y_scalar, train_data, val_data, plot_params['plot_bins'])
                                else:
                                    error_fig = None
                                    train_stats, val_stats = get_error_stats(model, train_loader, val_loader, y_columns[output_labels], y_scalar)
                                col = [stats_df.columns[x] for n1 in output_labels for x in range(4*n1,4*n1+4)]
                                stats_df.loc[iter,col] = get_stats_row(train_stats, val_stats)
                                loss_df.loc[iter,col] = get_loss_row(loss_func)

                                # Save the figures
                                loss_fig.savefig(BASE_NAME + LOSS_FOLDER + f"{iter}.png")
                                if plot_params['error_distribution']:
                                    error_fig.savefig(BASE_NAME + DIST_FOLDER + f"{iter}.png")
                                # Save the dataframes
                                stats_df.to_csv(BASE_NAME + DIST_FOLDER + RUN_NAME + f"_stats.csv")
                                loss_df.to_csv(BASE_NAME + LOSS_FOLDER + RUN_NAME + f"_loss.csv")
                                

                                # Update progress
                                iter += 1
                                pbar.update()

                            except Exception as e:
                                print(f"Error preparing model: {e}", flush=True)
                                pbar.update()
                                continue

                    except Exception as e:
                        print(f"Error preparing data: {e}", flush=True)
                        iter += JUMPS[1]
                        pbar.update(JUMPS[1])
                        continue

            except Exception as e:
                print(f"Error filtering data: {e}", flush=True)
                iter += JUMPS[0]
                pbar.update(JUMPS[0])
                continue

        # Save the model
        best_model.save(BASE_NAME + SAVED_MODELS_FOLDER + RUN_NAME + ".pth")

        # Save the report
        nan_columns = stats_df.columns[stats_df.isna().all()].tolist()
        stats_df.drop(columns=nan_columns, inplace=True)
        md["Error Figure"].add_image(plot_stats(stats_df[col]))

        nan_columns = loss_df.columns[loss_df.isna().all()].tolist()
        loss_df.drop(columns=nan_columns, inplace=True)
        md["Loss Values"].add_table(loss_df)
        md.save()
        print("*** Finished ***")

