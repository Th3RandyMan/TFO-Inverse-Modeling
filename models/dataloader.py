from copy import deepcopy
from typing import Dict, List, Optional, Tuple
from pandas import DataFrame
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from .validation_methods import ValidationMethod

class DataLoaderGenerator:
    """
    Class to generate DataLoader objects for training and validation data.
    """
    def __init__(
            self,
            data=None, 
            x_columns:List[str]=None,
            y_columns:List[str]=None,
            validation_method:ValidationMethod=None,
            batch_size:int=32,
            data_loader_params:Optional[Dict] = None,
            device: torch.device = None
            ):
        """
        Args:
            data: Data to be used for training and validation.
            batch_size (int): Batch size for DataLoader objects.
            data_loader_params (Dict): Additional parameters for DataLoader objects.
            device (torch.device): Device to be used for training.
        Returns:
            Tuple[DataLoader, DataLoader]: DataLoader objects for training and validation data.
        """
        self.data = data
        self.x_columns = x_columns
        self.y_columns = y_columns
        self.batch_size = batch_size

        if validation_method is not None:
            self.validation_method = validation_method
        else:
            from .validation_methods import RandomSplit
            self.validation_method = RandomSplit()

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        if data_loader_params is None:
            self.data_loader_params = {
            'shuffle': False,    # The dataloader will shuffle its outputs at each epoch
            'num_workers': 0,   # The number of workers that the dataloader will use to generate the batches
            'drop_last': True,  # Drop the last batch if it is smaller than the batch size
            }
        else:
            self.data_loader_params = deepcopy(data_loader_params)

    def change_batch_size(self, new_batch_size: int) -> None:
        """
        Changes the batch size of the dataloaders

        Args:
            new_batch_size (int): New batch size
        """
        self.data_loader_params["batch_size"] = new_batch_size
        self.batch_size = new_batch_size

    def generate(
            self, 
            data:DataFrame=None, 
            x_columns:List[str]=None,
            y_columns:List[str]=None,
            validation_method:ValidationMethod=None,
            batch_size:int=-1,
            data_loader_params:Optional[Dict] = None,
            device: torch.device = None
            ) -> Tuple[DataLoader, DataLoader]:
        """
        Generate DataLoader objects for training and validation data.
        Args:
            validation_method (ValidationMethod): Validation method to be used for splitting the data.
            data (DataFrame): Data to be used for training and validation.
            batch_size (int): Batch size for DataLoader objects.
            data_loader_params (Dict): Additional parameters for DataLoader objects.
            device (torch.device): Device to be used for training.
        Returns:
            Tuple[DataLoader, DataLoader]: DataLoader objects for training and validation data.
        """
        if data is not None:
            self.data = data
        else:
            if self.data is None:
                raise ValueError("data must be provided")

        if x_columns is not None:
            self.x_columns = x_columns
        else:
            if self.x_columns is None:
                raise ValueError("x_columns must be provided")
            
        if y_columns is not None:
            self.y_columns = y_columns
        else:
            if self.y_columns is None:
                raise ValueError("y_columns must be provided")
            
        if validation_method is not None:
            self.validation_method = validation_method
        if batch_size != -1:
            self.batch_size = batch_size
        if data_loader_params is not None:
            self.data_loader_params = deepcopy(data_loader_params)
        if device is not None:
            self.device = device

        # Get the training and validation data
        train_data, val_data = self.validation_method.split(self.data)

        # Create Datasets
        x_column_indices = [self.data.columns.tolist().index(x) for x in self.x_columns]
        y_column_indices = [self.data.columns.tolist().index(y) for y in self.y_columns]
        training_dataset = TensorDataset(
            torch.tensor(train_data.iloc[:, x_column_indices].values, dtype=torch.float32).cuda(),
            torch.tensor(train_data.iloc[:, y_column_indices].values, dtype=torch.float32).cuda(),
        )
        validation_dataset = TensorDataset(
            torch.tensor(val_data.iloc[:, x_column_indices].values, dtype=torch.float32).cuda(),
            torch.tensor(val_data.iloc[:, y_column_indices].values, dtype=torch.float32).cuda(),
        )

        # Create the data loaders
        train_loader = DataLoader(training_dataset, **self.data_loader_params)
        val_loader = DataLoader(validation_dataset, **self.data_loader_params)

        return train_loader, val_loader


if __name__ == "__main__":
    import numpy as np
    import pandas as pd

    #data = pd.DataFrame(np.array([range(100)]).T)
    # data = np.array(range(100))
    data = np.random.rand(100, 2)
    data_loader_generator = DataLoaderGenerator(data, batch_size=32)
    train_loader, val_loader = data_loader_generator.generate()
    print(train_loader)
    print(val_loader)