
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from .loss_functions import LossFunction, TorchLossWrapper, SumLoss
from sklearn.preprocessing import StandardScaler
from torch.nn import Module
from torch.utils.data import DataLoader
from pandas import DataFrame, Index
from typing import Tuple, Union, List
from .distributions import generate_model_error_and_prediction

"""
Functions for plotting the loss
"""
def _set_plot_settings(log:bool=False, title:str=None, xlabel:str=None, ylabel:str=None, legend:bool=False) -> None:
    """
    Set the plot settings

    Args:
        title: Title of the plot
        log: Whether to plot the loss on a logarithmic scale
        xlabel: Label for the x-axis
        ylabel: Label for the y-axis
        legend: Whether to include a legend
    """
    if log:
        plt.yscale('log')
    if title is not None:
        plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if legend:
        plt.legend()
    

def get_loss_fig(loss_func: LossFunction, log:bool=False, title:str=None, xlabel:str=None, ylabel:str=None, legend:bool=True) -> Figure:
    """
    Get the figure for the loss function

    Args:
        loss_func: Loss function
        log: Whether to plot the loss on a logarithmic scale
        title: Title of the plot
        xlabel: Label for the x-axis
        ylabel: Label for the y-axis
        legend: Whether to include a legend

    Returns:
        Figure object
    """

    if isinstance(loss_func, TorchLossWrapper):
        fig = plt.figure()
        loss_func.loss_tracker.plot_losses(legend=legend)
        _set_plot_settings(log, title, xlabel, ylabel)
    elif isinstance(loss_func, SumLoss):
        fig, axes = plt.subplots(1, len(loss_func.train_losses), squeeze=True, figsize=(3*len(loss_func.train_losses), 4), sharey=True)
        axes = axes.flatten()
        for i, loss_set in enumerate(zip(loss_func.train_losses, loss_func.val_losses)):
            plt.sca(axes[i])
            plt.title(loss_set[0].split('_')[0])
            for loss in loss_set:
                plt.plot(loss_func.loss_tracker.epoch_losses[loss], label='_'.join(loss.split('_')[-2:]))
            if i == 0:
                _set_plot_settings(log, None, xlabel, ylabel, legend)
            # else:
            #     _set_plot_settings(log, None, xlabel, ylabel)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


"""
Functions for plotting the error distribution plots
"""
def get_error_distrubition_fig(model:Module, train_loader:DataLoader, val_loader:DataLoader, y_columns:Union[Index, List[str]], y_scalar:StandardScaler, train_data:DataFrame=None, val_data:DataFrame=None, bin_count:int=50) -> Figure:
    """
    Get the figure for the error distribution plots

    Args:
        model: Model
        dataloader: DataLoader
        y_columns: Output columns
        y_scalar: Scaler for the output columns

    Returns:
        Figure object
    """
    train_error, train_pred = generate_model_error_and_prediction(model, train_loader, y_columns, y_scalar)
    val_error, val_pred = generate_model_error_and_prediction(model, val_loader, y_columns, y_scalar)

    if train_data is None:
        train_data = y_scalar.inverse_transform(train_loader.dataset[:][1].cpu())
    if val_data is None:
        val_data = y_scalar.inverse_transform(val_loader.dataset[:][1].cpu())

    fig, axes = plt.subplots(3, len(y_columns), figsize=(3*len(y_columns), 8), sharey=True)
    if len(axes.shape) == 1:    # If only one column
        axes = axes.reshape(3, 1)
    
    u_train_err = np.zeros(len(y_columns))
    std_train_err = np.zeros(len(y_columns))
    u_val_err = np.zeros(len(y_columns))
    std_val_err = np.zeros(len(y_columns))
    for i in range(len(y_columns)): # Can change range and bins by changing the range param in hist instead of axis
        xlim = [min(train_data[:, i].min(), val_data[:, i].min()), max(train_data[:, i].max(), val_data[:, i].max())]
        
        # Plot Errors
        ax = axes[0, i]
        plt.sca(ax)
        column_name = train_error.columns[i]
        weights = np.ones(max(len(train_error[column_name]), len(val_error[column_name]))) / (len(train_error[column_name]) + len(val_error[column_name]))
        plt.hist(train_error[column_name], bins=bin_count, color='blue', alpha=0.5, label='Train', weights=weights[:len(train_error[column_name])])
        plt.hist(val_error[column_name], bins=bin_count, color='orange', alpha=0.5, label='Validation', weights=weights[:len(val_error[column_name])])
        axes[0, i].set_xlim([0, max(train_error[column_name].max(), val_error[column_name].max())])
        
        u_train_err[i] = train_error[column_name].mean()
        std_train_err[i] = train_error[column_name].std()
        u_val_err[i] = val_error[column_name].mean()
        std_val_err[i] = val_error[column_name].std()

        # Plot Predictions
        ax = axes[1, i]
        plt.sca(ax)
        column_name = train_pred.columns[i]
        weights = np.ones(max(len(train_pred[column_name]), len(val_pred[column_name]))) / (len(train_pred[column_name]) + len(val_pred[column_name]))
        plt.hist(train_pred[column_name], bins=bin_count, color='blue', alpha=0.5, label='Train', weights=weights[:len(train_pred[column_name])])
        plt.hist(val_pred[column_name], bins=bin_count, color='orange', alpha=0.5, label='Validation', weights=weights[:len(val_pred[column_name])])
        axes[1, i].set_xlim(xlim)
        
        # Plot Ground Truth
        ax = axes[2, i]
        plt.sca(ax)
        weights = np.ones(max(len(train_data[:, i]), len(val_data))) / (len(train_data[:, i]) + len(val_data[:, i]))
        plt.hist(train_data[:, i], bins=bin_count, color='blue', alpha=0.5, label='Train', weights=weights[:len(train_data[:, i])])
        plt.hist(val_data[:, i], bins=bin_count, color='orange', alpha=0.5, label='Validation', weights=weights[:len(val_data[:, i])])
        axes[2, i].set_xlim(xlim)

        # X Label for the bottommost row
        plt.xlabel(y_columns[i])

    # Y Labels
    axes_labels = ['MAE Error', 'Prediction', 'Ground Truth']
    for i in range(axes.shape[0]):
        axes[i, 0].set_ylabel(axes_labels[i])

    # Add labels to top-left subplot
    axes[0, 0].legend()
    plt.tight_layout()

    return fig, (u_train_err, std_train_err), (u_val_err, std_val_err) 


def get_error_stats(model:Module, train_loader:DataLoader, val_loader:DataLoader, y_columns:Union[Index, List[str]], y_scalar:StandardScaler) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Get the error statistics

    Args:
        model: Model
        dataloader: DataLoader
        y_columns: Output columns
        y_scalar: Scaler for the output columns

    Returns:
        Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]: (u_train_err, std_train_err), (u_val_err, std_val_err)
    """
    train_error, _ = generate_model_error_and_prediction(model, train_loader, y_columns, y_scalar)
    val_error, _ = generate_model_error_and_prediction(model, val_loader, y_columns, y_scalar)

    u_train_err = np.zeros(len(y_columns))
    std_train_err = np.zeros(len(y_columns))
    u_val_err = np.zeros(len(y_columns))
    std_val_err = np.zeros(len(y_columns))
    for i in range(len(y_columns)):
        column_name = train_error.columns[i]
        u_train_err[i] = train_error[column_name].mean()
        std_train_err[i] = train_error[column_name].std()
        u_val_err[i] = val_error[column_name].mean()
        std_val_err[i] = val_error[column_name].std()

    return (u_train_err, std_train_err), (u_val_err, std_val_err)


class MetricTracker():
    """
    
    """
    def __init__(self):
        self.metrics = DataFrame()