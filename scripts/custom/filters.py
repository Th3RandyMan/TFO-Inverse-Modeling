from typing import List, Optional, Tuple, Union
import numpy as np
from pandas import DataFrame
from ...DLTools.processing.validation_methods import RandomSplit


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