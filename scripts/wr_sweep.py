from sweep import BaseSweeper
from DLTools.processing import RandomSplit, get_individual_criterion
import torch
import torch.nn as nn

import sys
sys.path.append(r'/home/rlfowler/Documents/research/TFO-Inverse-Modeling')


class DeltaSweeper(BaseSweeper):
    """
    Sweeper class used for Change Detection project.
    """
    def __init__(self, data_path: str, **kwargs) -> None:
        super().__init__(data_path, **kwargs)
        self.set_defaults()

    def set_defaults(self) -> None:
        """
        
        """
        # Constants
        self.TITLE = "Results for Change Detection"

        # Data Sweep
        self.data_sweep_dict["output_labels"] = ["Fetal Saturation"] # Change this
        self.data_sweep_dict["input_labels"] = [*range(20)] # Change this
        self.data_sweep_dict["filter_method"] = [None]
        self.data_sweep_dict['validation_method'] = [RandomSplit(0.8)] 
        self.data_sweep_dict["log_transform"] = [False]
        self.data_sweep_dict['batch_size'] = [64]
        self.data_sweep_dict['random_seed'] = [42]

        # Training Sweep
        self.train_sweep_dict['models'] = [*range(20)]   # Change this
        self.train_sweep_dict['num_epochs'] = [300]
        self.train_sweep_dict['learning_rate'] = [1e-4]
        self.train_sweep_dict['weighted_decay'] = [1e-4]
        self.train_sweep_dict['optimizer'] = [torch.optim.Adam]
        self.train_sweep_dict['loss_function'] = [nn.MSELoss]
        self.train_sweep_dict['loss_tracker'] = [get_individual_criterion]


if __name__ == "__main__":
    sweep = DeltaSweeper(r'/home/rlfowler/Documents/research/TFO-Simulator/data/weitai_epr.parquet')
    sweep.run()