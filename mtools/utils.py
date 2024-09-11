"""
Misc. functions useful during model training
"""
from typing import List, Optional, Tuple, Union
import numpy as np
from pandas import DataFrame, Index
from torch.nn.modules.loss import _Loss
from .loss_functions import LossFunction, SumLoss, TorchLossWrapper
from .validation_methods import CombineMethods, HoldOneOut, RandomSplit


def total_counter(*dictionaries: Tuple[dict]) -> int:
    """
    Counts the total number of combinations of elements in the dictionaries

    Args:
        dictionaries: Dictionaries to check

    Returns:
        Total number of combinations
    """
    return np.prod([len(dictionary[key]) for dictionary in dictionaries for key in dictionary])


class CombineFilters:
    """
    Combines multiple data filters
    """
    def __init__(self, filters: Union[List[callable],callable], *args) -> None:
        """
        Args:
            filters: List of data filters
        """
        if type(filters) == list:
            self.filters = filters
        elif callable(filters):
            self.filters = [filters]
        else:
            raise ValueError("filters must be a list or callable")
        
        for arg in args:
            if callable(arg):
                self.filters.append(arg)
            else:
                raise ValueError("args must be a callable")

    def __call__(self, *args) -> DataFrame:
        """
        Filters the data using the combined filters

        Args:
            data: Data to filter

        Returns:
            Filtered data
        """
        for filter in self.filters:
            data = filter(*args)
        return data

"""
Functions for filtering data
"""
def data_filter1(data:DataFrame, LABEL_START_INDEX=None) -> DataFrame:
    """
    Filters the data using a custom method.
    The data is filtered to keep only a subset of the unique values in the columns:
        - 'Fetal Hb Concentration': Every 3rd value starting from the 2nd
        - 'Fetal Radius': First 11 values
        - 'Maternal Saturation': Every 2nd value
        - 'Maternal Hb Concentration': Every 2nd value

    Args:
        data (DataFrame): Data to filter

    Returns:
        Filtered data (DataFrame)
    """
    columns = ['Fetal Hb Concentration', 'Fetal Radius', 'Maternal Saturation', 'Maternal Hb Concentration']
    to_keep = [np.sort(np.unique(data['Fetal Hb Concentration']))[1::3],\
                np.sort(np.unique(data['Fetal Radius']))[:11],\
                np.sort(np.unique(data['Maternal Saturation']))[::2],\
                np.sort(np.unique(data['Maternal Hb Concentration']))[::2]]
    for col, keep in zip(columns, to_keep):
        data = data.loc[data[col].isin(keep)]
    return data

def data_filter2(data:DataFrame, LABEL_START_INDEX=None) -> DataFrame:
    """
    Filters the data using a custom method.
    The data is filtered to keep only a subset of the unique values in the columns:
        - 'Fetal Hb Concentration': Every 3rd value starting from the 2nd
        - 'Fetal Radius': Last set of values (11th to last)
        - 'Maternal Saturation': Every 2nd value
        - 'Maternal Hb Concentration': Every 2nd value
        - 'Fetal Saturation': Every 2nd value

    Args:
        data (DataFrame): Data to filter

    Returns:
        Filtered data (DataFrame)
    """
    columns = ['Fetal Hb Concentration', 'Fetal Radius', 'Maternal Saturation', 'Maternal Hb Concentration','Fetal Saturation']
    to_keep = [np.sort(np.unique(data['Fetal Hb Concentration']))[1::3],\
                    np.sort(np.unique(data['Fetal Radius']))[11:],\
                    np.sort(np.unique(data['Maternal Saturation']))[::2],\
                    np.sort(np.unique(data['Maternal Hb Concentration']))[::2],\
                    np.sort(np.unique(data['Fetal Saturation']))[::2]\
                    ]
    for col, keep in zip(columns, to_keep):
        data = data.loc[data[col].isin(keep)]
    return data

def data_filter_remove_fr(data:DataFrame, LABEL_START_INDEX=None) -> DataFrame:
    """
    
    """
    LAST = 10   # Get last 10 values, hopefully the larger values

    if 'Fetal Radius' not in data.columns:
        raise ValueError("Fetal Radius column not found in data")
    
    uniq_fr = np.unique(data['Fetal Radius'])
    if len(uniq_fr) <= LAST:
        print(f"Unique Fetal Radius values less than or equal to {LAST}. Returning original data")
        return data

    fetal_radius_keep = np.unique(data['Fetal Radius'])[-1*LAST:]
    data = data[data['Fetal Radius'].isin(fetal_radius_keep)]
    
    return data

def data_filter_remove_fd(data:DataFrame, LABEL_START_INDEX=None) -> DataFrame:
    """
    
    """
    if 'Fetal Displacement' not in data.columns:
        raise ValueError("Fetal Displacement column not found in data")
    
    data = data[data['Fetal Displacement'].isin([0])]

    return data

# Dont use! Just select inputs
# def data_filter_5_detectors(data:DataFrame, LABEL_START_INDEX=None) -> DataFrame:
#     """
    
#     """
#     SELECTED_DISTANCES = [15, 33, 46, 68, 94]

#     if LABEL_START_INDEX is None:
#         for i, col in enumerate(data.columns):
#             col = col.split('_')[0]
#             try:    # Look for column with float values
#                 float(col)
#                 LABEL_START_INDEX = i
#                 break
#             except:
#                 continue

#     #wavelengths = []
#     distances = []

#     for det_name in data.columns[LABEL_START_INDEX:]:
#         name_split = det_name.split('_')
#         distances += [name_split[0]]
#         #wavelengths += [name_split[1]]

#     distances = list(set(float(dist) for dist in distances))
#     #wavelengths = list(set(wavelengths))
#     closest_distance = [str(min(distances, key=lambda x: abs(x - dist))) for dist in SELECTED_DISTANCES]

#     keep_list = data.columns[:LABEL_START_INDEX].tolist()
#     for det_name in data.columns[LABEL_START_INDEX:]:
#         name_split = det_name.split('_')
#         if name_split[0] in closest_distance:
#             keep_list.append(det_name)
    
#     return data[keep_list]
    




# def pr_5_detector_filter(data:DataFrame) -> DataFrame:
#     """
    
#     """
#     pass


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
    train_data, _ = RandomSplit(train_split=fraction).split(data) # Why reinvent the wheel?
    return train_data


"""
Functions for premade validation methods
"""
class custom_holdout(CombineMethods,HoldOneOut):
    """
    
    """
    def __init__(self, output_holds: Union[List[int], int], index_holds: Union[List[int], List[List[int]], int] = 3, random_split: Optional[Union[float, bool]] = False, label_start_index: int = 7):
        """
        Args:
            output_holds (List[int] | int): List of indices of output columns to hold out
            index_holds (List[int] | List[List[int]] | int): List of indices of unique values to hold out
            random_split (Optional[float | bool]): Fraction of data to hold out or whether to randomly split the data
            label_start_index (int): Index of the first input column
        """
        if type(output_holds) == int:
            output_holds = [output_holds]
        self.output_holds = output_holds
        
        if type(output_holds) == list:  # Multiple column holdout
            if type(index_holds) == int:    # Each column has the same holdout
                self.index_holds = [index_holds] * len(output_holds)
            elif len(index_holds) == 1:     # Each column has the same holdout
                self.index_holds = index_holds * len(output_holds)
            elif len(output_holds) != len(index_holds):   # Length of output_holds and index_holds must be equal
                raise ValueError("Length of output_holds and index_holds must be equal")
            else:   # Custom holdout for each column
                self.index_holds = index_holds
        else:
            raise ValueError("output_holds must be a list")
            
        self.random_split = random_split
        self.LABEL_START_INDEX = label_start_index

    def split(self, data:DataFrame) -> Tuple[DataFrame, DataFrame]:
        """
        Splits the data into training and validation data

        Args:
            data: Data to split

        Returns:
            Tuple of train and validation data
        """
        y_columns = data.columns[:self.LABEL_START_INDEX]
        if len(self.output_holds) == 1: # Single column holdout
            if self.random_split:
                CombineMethods.__init__(self, [HoldOneOut(y_columns[self.output_holds[0]], data[y_columns[self.output_holds[0]]].unique()[self.index_holds[0]]), RandomSplit(train_split=self.random_split)])
                return CombineMethods.split(self, data)
            else:
                HoldOneOut.__init__(self, y_columns[self.output_holds[0]], data[y_columns[self.output_holds[0]]].unique()[self.index_holds[0]])
                return HoldOneOut.split(self, data)
        else:   # Multiple column holdout
            val_methods = [HoldOneOut(y_columns[output_hold], data[y_columns[output_hold]].unique()[index_hold]) for output_hold, index_hold in zip(self.output_holds, self.index_holds)]
            if self.random_split:
                val_methods.append(RandomSplit(train_split=self.random_split))
            CombineMethods.__init__(self, val_methods)
            return CombineMethods.split(self, data)
        

"""
Functions for premade criterion functions
"""
def get_combined_criterion(loss_func: _Loss, y_columns:Union[Index, List[str]] = None, output_labels: Optional[Union[List[int], int]]=None, name:str=None) -> LossFunction:
    """
    Get the loss function wrapper based on the loss function and output labels.

    Args:
        loss_func: Loss function

    Returns:
        Loss function
    """
    return TorchLossWrapper(loss_func(), name=name)

def get_individual_criterion(loss_func: _Loss, y_columns:Union[Index, List[str]], output_labels: Optional[Union[List[int], int]]=None, name:str=None) -> LossFunction:
    """
    Get the loss function wrapper based on the loss function and output labels.

    Args:
        loss_func: Loss function
        y_columns: Output columns
        output_labels: Output labels

    Returns:
        Loss function
    """
    if output_labels is None:
        output_labels = [*range(len(y_columns))]
    elif type(output_labels) == int:
        output_labels = [output_labels]
    return SumLoss([TorchLossWrapper(loss_func(), [i], y_columns[i]) for i in range(len(y_columns)) if i in output_labels], weights=[1/len(output_labels)]*len(output_labels))