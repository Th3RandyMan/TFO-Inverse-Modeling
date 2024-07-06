
from typing import List, Optional
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from .validation_methods import ValidationMethod
from .loss_functions import LossFunction
from .misc import DATA_LOADER_INPUT_INDEX
from torch.optim import Optimizer

class BaseModel(nn.Module):
    """
    Base class for all models.
    """
    model: nn.Module    # This cannot be None. It must be set in the child class
    layers: List[nn.Module] = []

    def __init__(self, validation_method: ValidationMethod=None, loss_func: LossFunction=None) -> None:
        super().__init__()
        self.validation_method:ValidationMethod = validation_method
        self.loss_func:LossFunction = loss_func

    def _reset_layer(self, layer: nn.Module) -> None:
        """
        Reset the parameters of the layer.
        """
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()

    def reset_model(self) -> None:
        """
        Reset the model parameters
        """
        self.model.apply(self._reset_layer)
        self.loss_func.reset()

    def reset_optimizer(self) -> None:
        """
        Reset the optimizer parameters
        """
        optim_type = type(self.optimizer)
        self.optimizer = optim_type(self.model.parameters(), **self.optimizer.defaults)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def run_training(self, optimizer: Optimizer, train_loader: DataLoader, val_loader: DataLoader=None, epochs: int=1, device: torch.device=None):
        """
        Run the training loop for the model
        Args:
            train_loader (DataLoader): DataLoader object for training data
            val_loader (DataLoader): DataLoader object for validation data
            epochs (int): Number of epochs for training
            device (torch.device): Device to be used for training
        """
        self.optimizer = optimizer
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        self.to(device)
        self.train()

        for epoch in range(epochs):
            # Training loop
            for data in train_loader:
                inputs = data[DATA_LOADER_INPUT_INDEX]

                # zero the parameter gradients - previous batch gradients are not used
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = self.loss_func(outputs, data, "train")
                loss.backward()
                self.optimizer.step()

            # Validation loop
            if val_loader is not None:
                self.eval()
                with torch.no_grad():
                    for data in val_loader:
                        inputs = data[DATA_LOADER_INPUT_INDEX]
                        outputs = self.model(inputs)
                        loss = self.loss_func(outputs, data, "validate")

                # Switch back to training mode
                self.train()
            self.loss_func.loss_tracker_epoch_update()

    def __str__(self) -> str:
        return f"""
        Model Properties:
        {self.model}
        Validation Method:
        {self.validation_method}
        Loss Function:
        {self.loss_func}
        Optimizer Properties":
        {self.optimizer}
        """


class MLP(BaseModel):
    """
    Multi Layer Perceptron model
    """
    def __init__(
            self, node_counts: List[int], 
            dropout_rates: Optional[List[float]] = None, 
            batch_norm: bool = True, 
            act_funcs: Optional[List[float]] = [nn.ReLU()],
            validation_method: ValidationMethod=None, 
            loss_func: LossFunction=None, 
            ) -> None:
        """
        Args:
            node_counts (List[int]): Number of nodes in each layer
            dropout_rates (Optional[List[float]]): Dropout rates for each layer
                - If None, dropout is not used
                - If a single value, the same dropout rate is used for all layers
            batch_norm (bool): Whether to use batch normalization
            act_funcs (Optional[List[nn.Module]]): Activation functions for each layer
                - If None, ReLU is used for all layers except the last layer
                - If a single value, the same activation function is used for all layers except the last layer
                - List can be equal to the number of layers or one less than the number of layers
            validation_method (ValidationMethod): Validation method
            loss_func (LossFunction): Loss function
            optimizer (Optimizer): Optimizer
        """
        super().__init__(validation_method, loss_func)
        
        dropout = False if dropout_rates is None else True
        if dropout and len(dropout_rates) != len(node_counts) - 1:
            if len(dropout_rates) == 1:
                dropout_rates = dropout_rates * (len(node_counts) - 1)
            else:
                raise ValueError("Dropout rates must be equal to the number of hidden layers or 1 for all layers.")
        if len(act_funcs) != len(node_counts) - 1:  # Last layer may or may not have an activation function
            if len(act_funcs) == 1:
                act_funcs = act_funcs * (len(node_counts) - 1)  # Assume last layer does not have an activation function
            elif len(act_funcs) != len(node_counts) - 2:    # Last layer may or may not have an activation function
                raise ValueError("Activation functions must be equal to the number of hidden layers or 1 for all layers.")

        self.layers = []    # Need this to reset layers when resetting the model
        for indx, node_count in enumerate(node_counts[:-2]):
            self.layers.append(nn.Linear(node_count, node_counts[indx + 1]))
            if batch_norm:
                self.layers.append(nn.BatchNorm1d(node_counts[indx + 1]))
            if dropout:
                self.layers.append(nn.Dropout(dropout_rates[indx]))
            self.layers.append(act_funcs[indx])
        
        self.layers.append(nn.Linear(node_counts[-2], node_counts[-1]))
        self.layers.append(nn.Flatten())
        if len(act_funcs) == len(node_counts):  # Last layer has an activation function
            self.layers.append(act_funcs[-1])
        
        # Create the model
        self.model = nn.Sequential(*self.layers)


