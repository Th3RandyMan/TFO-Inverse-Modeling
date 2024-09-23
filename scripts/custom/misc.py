from typing import Tuple
import numpy as np


def total_counter(*dictionaries: Tuple[dict]) -> int:
    """
    Counts the total number of combinations of elements in the dictionaries

    Args:
        dictionaries: Dictionaries to check

    Returns:
        Total number of combinations
    """
    return np.prod([len(dictionary[key]) for dictionary in dictionaries for key in dictionary])