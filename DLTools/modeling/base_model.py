from typing import List, Optional, Union
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from ..processing import DATA_LOADER_INPUT_INDEX
from ..processing.validation_methods import ValidationMethod
from ..processing.loss_functions import LossFunction
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
    
    def run_training(self, optimizer: Optimizer, train_loader: DataLoader, val_loader: DataLoader=None, epochs: int=1, device: torch.device=None, early_stop:Optional[int]=15, verbose:bool=False) -> float:
        """
        Run the training loop for the model
        Args:
            train_loader (DataLoader): DataLoader object for training data
            val_loader (DataLoader): DataLoader object for validation data
            epochs (int): Number of epochs for training
            device (torch.device): Device to be used for training

        Returns:
            float: Best validation loss
        """
        self.optimizer = optimizer
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        early_stop_counter = 0
        best_state = None
        best_loss = float("inf")

        self.to(device)
        self.train()

        for epoch in range(epochs):
            # Training loop
            for data in train_loader:
                inputs = data[DATA_LOADER_INPUT_INDEX] # Could be done here though
                inputs = inputs.to(device) # Slightly faster having this?

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
                # avg_loss = 0
                with torch.no_grad():
                    for data in val_loader:
                        inputs = data[DATA_LOADER_INPUT_INDEX]
                        outputs = self.model(inputs)
                        loss = self.loss_func(outputs, data, "validate")
                        # avg_loss += loss.item()

                # Switch back to training mode
                self.train()

                # Update loss tracker
                self.loss_func.loss_tracker_epoch_update()  
                # Should we add best_train_loss and best_val_loss for early stopping?
                avg_loss = self.loss_func.loss_tracker.epoch_losses['val_loss'][-1]                  
                
                # Print update message
                if verbose:
                    print(f"Epoch {epoch + 1}/{epochs} - Training Loss: {self.loss_func.loss_tracker.epoch_losses['train_loss'][-1]:.4e} - Validation Loss: {avg_loss:.4e}", flush=True)

                # Check validation loss for early stopping
                if early_stop is not None:  # MAYBE ADD CONDITION FOR EVERY LOSS TERM IN LOSS TRACKER
                    if avg_loss < best_loss:
                        best_loss = avg_loss
                        best_state = self.model.state_dict()
                        early_stop_counter = 0
                    else:
                        early_stop_counter += 1
                        if early_stop_counter == early_stop:
                            print(f"Early stopping at epoch {epoch}.")
                            break

            else:
                self.loss_func.loss_tracker_epoch_update()

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return best_loss

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
    
    def save(self, path: str) -> None:
        """
        Save the model to the specified path
        """
        if not path.endswith(".pth"):
            path += ".pth"

        import os
        
        parent_folder = os.path.dirname(path)
        if not os.path.exists(parent_folder):
            os.makedirs(parent_folder)

        torch.save(self.model.state_dict(), path)

    def load(self, path: str) -> None:
        """
        Load the model from the specified path
        """
        state_dict = torch.load(path)#, weight_only=True)
        
        try:
            self.model.load_state_dict(state_dict)#, weight_only=True)

        except RuntimeError as e:
            raise RuntimeError(f"Check if keys are off. Error: {e}")