"""
Misc. functions useful during model training
"""
import random
from typing import Tuple
import numpy as np
from pandas import DataFrame
import torch

# CONSTANTS
DATA_LOADER_INPUT_INDEX, DATA_LOADER_LABEL_INDEX, DATA_LOADER_EXTRA_INDEX = 0, 1, 2

def set_seed(seed):
    """Manually sets the seed for python internals, numpy, torch and CUDA

    Args:
        seed: SEED
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def total_counter(*dictionaries: Tuple[dict]) -> int:
    """
    Counts the total number of combinations of elements in the dictionaries

    Args:
        dictionaries: Dictionaries to check

    Returns:
        Total number of combinations
    """
    return np.prod([len(dictionary[key]) for dictionary in dictionaries for key in dictionary])

"""
Functions for filtering data
"""
def data_filter1(data:DataFrame) -> DataFrame:
    """
    Filters the data using the custom method.

    Args:
        data: Data to filter

    Returns:
        Filtered data
    """
    raise NotImplementedError("Custom data filter not implemented")
    return data


def random_filter(data:DataFrame, fraction: float = 0.05) -> DataFrame:
    """
    Filters the data randomly

    Args:
        data: Data to filter
        fraction: Fraction of data to keep
            - Default: 0.05, 5% of the data is kept

    Returns:
        Filtered data
    """
    from .validation_methods import RandomSplit # Why reinvent the wheel?
    train_data, _ = RandomSplit(train_split=fraction).split(data)
    return train_data


"""
Functions for premade validation methods
"""
def validation1(data):
    """
    Custom validation method

    Args:
        data: Data to split

    Returns:
        Tuple of train and validation data
    """
    raise NotImplementedError("Custom validation method not implemented")
    return data, data