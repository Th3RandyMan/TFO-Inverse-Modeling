
from collections import defaultdict
# Replace later when package is established
import itertools
import os
# import sys
from typing import Tuple
from PyMD import MDGenerator
import numpy as np
import pandas as pd
from sklearn import preprocessing
from tqdm import tqdm

from DLTools.modeling.base_model import BaseModel
from DLTools.processing import DataLoaderGenerator
# sys.path.append(r'/home/rlfowler/Documents/research/TFO-Inverse-Modeling')

class BaseSweeper:
    """
    
    """
    # Report settings
    RUN_NAME:str = "default_run"
    TITLE:str = "Default Title"
    DATA_PATH:str = None
    BASE_FOLDER:str="reports/"
    LOSS_FOLDER = "losses/"
    DIST_FOLDER = "distributions/"
    SAVED_MODELS_FOLDER = "saved_models/"
    SAVE_IN_BASE_FOLDER:bool = False

    # Sweep jump settings
    SKIP_TO = 0
    SKIP_LIST = []

    # Data Handling settings
    GPU_ID:int = 2 # GPU 2 - Randall
    COPY_DATA:bool=True
    DATA_LOADER_PARAMS = None
    LABEL_NAMES:list = []
    LABEL_START_INDEX:int = None

    # Training settings
    EARLY_STOP:int = 15
    VERBOSE:bool = True

    # Default data parameters
    data_sweep_dict = defaultdict(list)
    data_sweep_dict["output_labels"] = []
    data_sweep_dict["input_labels"] = []
    data_sweep_dict["filter_method"] = []
    data_sweep_dict['validation_method'] = [] 
    data_sweep_dict["log_transform"] = [False]
    data_sweep_dict['batch_size'] = []
    data_sweep_dict['random_seed'] = []

    # Default training parameters
    train_sweep_dict = defaultdict(list)
    train_sweep_dict['models'] = []
    train_sweep_dict['num_epochs'] = []
    train_sweep_dict['learning_rate'] = []
    train_sweep_dict['weighted_decay'] = []
    train_sweep_dict['optimizer'] = []
    train_sweep_dict['loss_function'] = []
    train_sweep_dict['loss_tracker'] = []

    # Default plotting settings
    plot_settings = {}
    plot_settings['loss'] = {}                              # Loss plot settings  
    plot_settings['loss']["Use"] = True                         # Use the plot
    plot_settings['loss']["title"] = "Loss"                     # Title of the plot
    plot_settings['loss']["xlabel"] = "Epoch"                   # X-axis label
    plot_settings['loss']["ylabel"] = None                      # Y-axis label
    plot_settings['loss']["legend"] = True                      # Use legend

    plot_settings['error'] = {}                             # Error plot settings
    plot_settings['error']["Use"] = True                        # Use the plot  
    plot_settings['error']["Use R Value"] = True                # Use R value
    plot_settings['error']["title"] = "Error"                   # Title of the plot
    plot_settings['error']["xlabel"] = "Epoch"                  # X-axis label
    plot_settings['error']["ylabel"] = None                     # Y-axis label
    plot_settings['error']["legend"] = True                     # Use legend

    plot_settings['distribution'] = {}                      # Distribution plot settings
    plot_settings['distribution']["Use"] = True                 # Use the plot
    plot_settings['distribution']["Use Error"] = True           # Use error
    plot_settings['distribution']["Use Prediction"] = True      # Use prediction
    plot_settings['distribution']["Use Ground Truth"] = True    # Use ground truth
    plot_settings['distribution']["title"] = "Distribution"     # Title of the plot
    plot_settings['distribution']["resolution"] = 4096          # Batch size for creating the plot
    plot_settings['distribution']['plot_bins'] = None           # Number of bins to use in the histogram

    def __init__(self, data_path:str, **kwargs) -> None:
        self.DATA_PATH = data_path
        self.update(**kwargs)        

    def update(self, **kwargs) -> None:
        """
        
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            elif key in self.plot_settings:
                if isinstance(value, dict):
                    for k, v in value.items():
                        if k in self.plot_settings[key]:
                            self.plot_settings[key][k] = v
                        else:
                            raise ValueError(f"Invalid key: {k}")
                elif isinstance(value, bool):
                    self.plot_settings[key]["Use"] = value
                else:
                    raise ValueError(f"Value must be a dictionary for configuring plot settings")
            else:
                if not isinstance(value, list) or key in ["output_labels", "input_labels"]:
                    value = [value]

                if key in self.data_sweep_dict:
                    for v in value:
                        self.data_sweep_dict[key].append(v)
                elif key in self.train_sweep_dict:
                    for v in value:
                        self.train_sweep_dict[key].append(v)
                else:
                    raise ValueError(f"Not a valid parameter: {key}")

    def run(self, **kwargs) -> None:
        """
        Run a sweep with the given parameters. Default parameters are used if not provided.

        Recommended Args:
            RUN_NAME: Name of the run
            TITLE: Title of the run
        """
        self.update(**kwargs)
        self._validate()
        self._create_folders()
        self._update_run_name()
        self._data_config()
        self._run_sweep()

    def _validate(self) -> None:
        """
        Function that checks if the parameters are valid.
        - Go through each dictionary and check if values are not empty.
        - Go through input and output labels and check if they are lists of lists, integers, or strings.
        """
        # Check if sweeping dictionaries are empty
        for dictionary in [self.data_sweep_dict, self.train_sweep_dict]:
            for key, value in self.data_sweep_dict.items():
                if not isinstance(value, list):
                    dictionary[key] = [value]
                    # if isinstance(value, np.array):
                    #     dictionary[key] = value.tolist()
                
                if len(value) == 0:
                    raise ValueError(f"Empty list for {key}")
        
        # Check if output_labels is a list of lists, integers, or strings
        if not all(isinstance(label, list) for label in self.data_sweep_dict['output_labels']):
            if all(isinstance(label, int) for label in self.data_sweep_dict['output_labels']):
                self.data_sweep_dict['output_labels'] = [self.data_sweep_dict['output_labels']]
            elif all(isinstance(label, str) for label in self.data_sweep_dict['output_labels']):
                self.data_sweep_dict['output_labels'] = [self.data_sweep_dict['output_labels']]
            else:
                raise ValueError("Invalid output_labels. Must be list of lists, integers, or strings.")
            
        # Check if input_labels is a list of lists, integers, or strings
        if not all(isinstance(label, list) for label in self.data_sweep_dict['input_labels']):
            if all(isinstance(label, int) for label in self.data_sweep_dict['input_labels']):
                self.data_sweep_dict['input_labels'] = [self.data_sweep_dict['input_labels']]
            elif all(isinstance(label, str) for label in self.data_sweep_dict['input_labels']):
                self.data_sweep_dict['input_labels'] = [self.data_sweep_dict['input_labels']]
            else:
                raise ValueError("Invalid input_labels. Must be list of lists, integers, or strings.")

        # Check that learning rate and weighted decay are not None and at least 0
        if any([value is None or value < 0 for value in self.train_sweep_dict['learning_rate']]):
            raise ValueError("Invalid learning rate")
        for value in self.train_sweep_dict['weighted_decay']:
            if value is None:
                value = 0
            elif value < 0:
                raise ValueError("Invalid weighted decay")

        # Add additional checks here

    def _create_folders(self) -> None:
        """
        Creates the necessary folders for the sweep.
        - If SAVE_IN_BASE_FOLDER is True, the folders are created in the BASE_FOLDER.
        - If SAVE_IN_BASE_FOLDER is False, the run name becomes a folder within the BASE_FOLDER.
        """
        if not self.SAVE_IN_BASE_FOLDER:
            self.BASE_FOLDER = self.BASE_FOLDER + self.RUN_NAME + "/"
        self.LOSS_FOLDER = self.BASE_FOLDER + self.LOSS_FOLDER
        self.DIST_FOLDER = self.BASE_FOLDER + self.DIST_FOLDER
        self.SAVED_MODELS_FOLDER = self.BASE_FOLDER + self.SAVED_MODELS_FOLDER

        if not os.path.exists(self.BASE_FOLDER):
            os.makedirs(self.BASE_FOLDER)
        if not os.path.exists(self.LOSS_FOLDER):
            os.makedirs(self.LOSS_FOLDER)
        if not os.path.exists(self.DIST_FOLDER):
            os.makedirs(self.DIST_FOLDER)
        if not os.path.exists(self.SAVED_MODELS_FOLDER):
            os.makedirs(self.SAVED_MODELS_FOLDER)

    def _update_run_name(self) -> None:
        """
        Check to see if the run name already exists and update it if it does. Enumerates the run name.
        """
        pass
        #raise NotImplementedError("Need to check if folder or file. Depending on self.SAVE_IN_BASE_FOLDER.")
        # if os.path.exists(self.BASE_FOLDER + self.RUN_NAME):
        #     i = 0
        #     while os.path.exists(self.BASE_FOLDER + self.RUN_NAME + f"_{i}"):
        #         i += 1
        #     self.RUN_NAME = self.RUN_NAME + f"_{i}"
    
    def _read_data(self) -> pd.DataFrame:
        """
        Method to read the data from the file path.
        - Will read the data based on the file type.
            - Only supports .pkl and .parquet files.
        """
        if self.DATA_PATH.endswith('.pkl'):
            return pd.read_pickle(self.DATA_PATH)
        elif self.DATA_PATH.endswith('.parquet'):
            return pd.read_parquet(self.DATA_PATH)
        else:
            raise ValueError(f"Invalid file type: {self.DATA_PATH}")

    def _data_config(self) -> None:
        """
        Configures the data settings for the sweep
        """
        # Set GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.GPU_ID)

        self._data = self._read_data()
        
        # Set LABEL_START_INDEX and LABEL_NAMES
        if self.LABEL_START_INDEX is None or self.LABEL_START_INDEX < 0:  # If LABEL_START_INDEX is not set
            if self.LABEL_NAMES and isinstance(self.LABEL_NAMES, list) and len(self.LABEL_NAMES) > 0:
                self.LABEL_START_INDEX = len(self.LABEL_NAMES) if self.LABEL_NAMES else None
            else:   # Find names and index of labels
                columns = self._data.columns

                if self.LABEL_START_INDEX is None:
                    for i, col in enumerate(columns):
                        col = col.split('_')[0]
                        try:    # Look for column with float values
                            float(col)
                            self.LABEL_START_INDEX = i
                            break
                        except:
                            continue
                
                self.LABEL_NAMES = columns[:self.LABEL_START_INDEX]
        else:   # If LABEL_START_INDEX is set
            if self.LABEL_NAMES and isinstance(self.LABEL_NAMES, list) and len(self.LABEL_NAMES) > 0:
                self.LABEL_START_INDEX = len(self.LABEL_NAMES)
            else:
                self.LABEL_NAMES = self._data.columns[:self.LABEL_START_INDEX]

        if not isinstance(self.LABEL_NAMES, list):
            self.LABEL_NAMES = self.LABEL_NAMES.tolist()
        self.INPUT_LABEL_NAMES = self._data.columns[self.LABEL_START_INDEX:].to_list()

        # Update output labels if string (prevalidated)
        for i, output_labels in enumerate(self.data_sweep_dict['output_labels']):
            for j, label in enumerate(output_labels):
                if isinstance(label, str):
                    if label in self.LABEL_NAMES:
                        self.data_sweep_dict['output_labels'][i][j] = self.LABEL_NAMES.index(label)
                    else:
                        raise ValueError(f"Invalid output label: {label}")
                else:
                    if label >= self.LABEL_START_INDEX or label < 0:
                        raise ValueError(f"Invalid output label index: {label}")
                    
        # Update input labels if string (prevalidated)
        for i, input_labels in enumerate(self.data_sweep_dict['input_labels']):
            for j, label in enumerate(input_labels):
                if isinstance(label, str):
                    if label in self.INPUT_LABEL_NAMES:
                        self.data_sweep_dict['input_labels'][i][j] = self.INPUT_LABEL_NAMES.index(label)
                    else:
                        raise ValueError(f"Invalid input label: {label}")
                else:
                    if label >= len(self.INPUT_LABEL_NAMES) or label < 0:
                        raise ValueError(f"Invalid input label index: {label}")
                    
        # Add additional data configuration here

    def _total_counter(self, *dictionaries: Tuple[dict]) -> int:
        return np.prod([len(dictionary[key]) for dictionary in dictionaries for key in dictionary])

    def _handle_data(self, data:pd.DataFrame, filter_method, apply_log:bool) -> pd.DataFrame:
        """
        Handles the data by filtering, applying log transformation, and normalizing the data.

        Args:
            data (DataFrame): Data to handle
            filter_method (Callable): Method to filter the data
            apply_log (bool): Whether to apply log transformation

        Returns:
            Handled data
        """
        # Filter the data
        if filter_method is not None:
            data = filter_method(data)

        # Get input and output columns
        x_columns = data.columns[self.LABEL_START_INDEX:]    # Input columns
        y_columns = data.columns[:self.LABEL_START_INDEX]    # Output columns

        # Apply log transformation to the data
        if apply_log:
            data[x_columns] = np.log(data[x_columns])

        # Normalize the data
        y_scalar = preprocessing.StandardScaler()
        data[y_columns] = y_scalar.fit_transform(data[y_columns])
        x_scalar = preprocessing.StandardScaler()
        data[x_columns] = x_scalar.fit_transform(data[x_columns])

        return data
    
    def _train(self, train_loader:DataLoaderGenerator, val_loader:DataLoaderGenerator, **train_params) -> Tuple[object, float]:
        """
        Trains the model

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            **train_params: Parameters for training

        Returns:
            Model and loss
        """
        x_columns = self._data.columns[self.LABEL_START_INDEX:]    # Input columns
        y_columns = self._data.columns[:self.LABEL_START_INDEX]    # Output columns

        if train_params['loss_tracker'] is not None:
            loss_function = train_params['loss_tracker'](train_params['loss_function'], y_columns[train_params['output_labels']])
        else:
            loss_function = train_params['loss_function']

        model = train_params['models'](len(x_columns[train_params['input_labels']]), len(y_columns[train_params['output_labels']]), loss_function)
        optimizer = train_params['optimizer'](model.parameters(), lr=train_params['learning_rate'], weight_decay=train_params['weighted_decay'])
        val_loss = model.train(optimizer, train_loader, val_loader, train_params['num_epochs'], early_stop=self.EARLY_STOP, verbose=self.VERBOSE)

        return model, val_loss

    def _prepare_data(self, data:pd.DataFrame, **data_params) -> Tuple[DataLoaderGenerator, DataLoaderGenerator]:
        """
        Prepares the data for training

        Args:
            data: Data to prepare
            **kwargs: Parameters for DataLoaderGenerator

        Returns:
            Training and Validation data loaders
        """
        # raise NotImplementedError

        # Get columns
        x_columns = data.columns[self.LABEL_START_INDEX:]    # Input columns
        y_columns = data.columns[:self.LABEL_START_INDEX]    # Output columns

        # Get the data loader
        DLG = DataLoaderGenerator(data, x_columns[data_params['input_labels']], y_columns[data_params['output_labels']], data_params['validation_method'], data_params['batch_size'], self.DATA_LOADER_PARAMS)
        train_loader, val_loader = DLG.generate()

        return train_loader, val_loader

    def _run_sweep(self) -> None:
        """
        Runs the sweep
        """
        total = self._total_counter(self.data_sweep_dict, self.train_sweep_dict)
        if total == 0:
            raise ValueError("No sweeps to run")

        JUMP = [
            total // (len(self.data_sweep_dict['filter_method']) * len(self.data_sweep_dict['log_transform'])),
            self._total_counter(self.train_sweep_dict)
        ]

        iter:int = 0
        best_loss = np.inf
        best_model:BaseModel = None
        best_params = {}

        with tqdm(total=total, desc="Sweeping") as pbar:
            if self.COPY_DATA:
                df = self._data.copy()
            for filter_method, apply_log in itertools.product(self.data_sweep_dict['filter_method'], self.data_sweep_dict['log_transform']):
                if iter + JUMP[0] < self.SKIP_TO:
                    iter += JUMP[0]
                    pbar.update(JUMP[0])
                    continue

                try:
                    if self.COPY_DATA:
                        data = df.copy()
                    else:
                        data:pd.DataFrame = self._read_data()

                    # Handle the data
                    data = self._handle_data(data, filter_method, apply_log)

                    for data_params in itertools.product(*[self.data_sweep_dict[key] for key in self.data_sweep_dict.keys() if key != 'filter_method' and key != 'log_transform']):
                        if iter + JUMP[1] < self.SKIP_TO:
                            iter += JUMP[1]
                            pbar.update(JUMP[1])
                            continue

                        try:
                            data_params = dict(zip([key for key in self.data_sweep_dict.keys() if key != 'filter_method' and key != 'log_transform'], data_params))
                            
                            # Get the data loader
                            train_loader, val_loader = self._prepare_data(data, **data_params)

                            for train_params in itertools.product(*[self.train_sweep_dict[key] for key in self.train_sweep_dict.keys()]):
                                if iter in self.SKIP_LIST:
                                    iter += 1
                                    pbar.update(1)
                                    continue

                                try:
                                    train_params = dict(zip(self.train_sweep_dict.keys(), train_params))
                                    train_params['output_labels'] = data_params['output_labels']
                                    train_params['input_labels'] = data_params['input_labels']
                                    model, loss = self._train(train_loader, val_loader, **train_params)

                                    if loss < best_loss:
                                        best_loss = loss
                                        best_model = model
                                        best_params['data_params'] = data_params.update({'filter_method':filter_method, 'log_transform':apply_log})
                                        best_params['train_params'] = train_params

                                except Exception as e:
                                    print(f"Error training model: {e}")
                                    iter += 1
                                    pbar.update(1)
                                    continue
                        except Exception as e:
                            print(f"Error preparing data: {e}")
                            iter += JUMP[1]
                            pbar.update(JUMP[1])
                            continue
                except Exception as e:
                    print(f"Error filtering data: {e}")
                    iter += JUMP[0]
                    pbar.update(JUMP[0])
                    continue

        if best_model is not None:
            best_model.save(self.SAVED_MODELS_FOLDER + self.RUN_NAME + ".pth")
        print(f"Best Loss: {best_loss}")
        # Save info by saving dict of best

        


