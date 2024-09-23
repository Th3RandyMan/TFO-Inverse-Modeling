from .base_model import BaseModel
from ..processing.validation_methods import ValidationMethod
from ..processing.loss_functions import LossFunction
from typing import List, Optional, Tuple, Union
import torch.nn as nn

class MLP(BaseModel):
    """
    Multi Layer Perceptron model
    """
    def __init__(
            self, node_counts: List[int], 
            dropout_rates: Optional[Union[List[float], float]] = None, 
            batch_norm: bool = True, 
            act_funcs: Union[List[nn.Module], nn.Module] = [nn.ReLU()],
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
        
        dropout = False if dropout_rates is None or not dropout_rates or dropout_rates == 0 else True
        if dropout:
            if isinstance(dropout_rates, float):
                if dropout_rates < 0 or dropout_rates > 1:
                    raise ValueError("Dropout rate must be between 0 and 1.")
                dropout_rates = [dropout_rates] * (len(node_counts) - 1)
            elif len(dropout_rates) != len(node_counts) - 1:
                if len(dropout_rates) == 1:
                    if dropout_rates[0] < 0 or dropout_rates[0] > 1:
                        raise ValueError("Dropout rate must be between 0 and 1.")
                    dropout_rates = dropout_rates * (len(node_counts) - 1)
                else:
                    raise ValueError("Dropout rates must be equal to the number of hidden layers or 1 for all layers.")
        if isinstance(act_funcs, nn.Module):
            act_funcs = [act_funcs] * (len(node_counts) - 1)
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

class FC(MLP):
    def __init__(self, node_counts: List[int], dropout_rates: Optional[Union[List[float], float]] = None, batch_norm: bool = True, act_funcs: Union[List[nn.Module], nn.Module] = [nn.ReLU()], validation_method: ValidationMethod=None, loss_func: LossFunction=None) -> None:
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
        super().__init__(node_counts, dropout_rates, batch_norm, act_funcs, validation_method, loss_func)


class CNN(BaseModel):
    """
    Convolutional Neural Network model
    """
    def __init__(
        self, 
        channels: List[int],
        kernel_sizes: List[int],
        strides: Union[List[int], int, List[Tuple[int,int]], tuple] = 1,
        paddings: Union[List[int], int, str, List[str], List[tuple], tuple] = 0,
        dilations: Union[List[int], int, List[tuple], tuple] = 1,
        padding_modes: Union[List[str], str] = "zeros",
        groups: Union[List[int], int] = 1,
        cnn_bias: Union[List[bool], bool] = True,
        cnn_batch_norm: Optional[Union[List[bool], bool]] = False,
        cnn_act_funcs: Optional[Union[List[nn.Module], nn.Module]] = None,
        pooling: Optional[Union[List[str], str, nn.Module, List[nn.Module]]] = None,
        pooling_kernel_sizes: Union[List[int], int] = None,
        pooling_strides: Union[List[int], int] = None,
        pooling_paddings: Union[List[int], int] = None,
        pooling_dilations: Union[List[int], int] = None,
        ceil_modes: Union[List[bool], bool] = False,
        conv_dim: int = 1,
        flatten: bool = True,

        FC: Optional[MLP] = None,
        node_counts: Optional[List[int]] = None,
        dropout_rates: Optional[Union[List[float], float]] = None,
        FC_batch_norm: bool = False,
        FC_act_funcs: Optional[List[nn.Module]] = [nn.ReLU()],
        # FC_create_input_count: bool = False,

        validation_method: ValidationMethod=None,
        loss_func: LossFunction=None,
        ) -> None:
        """
        Args:
            channels (List[int]): Number of channels in each convolutional layer.
                - The first value is the number of input channels. The last value is the number of output channels.
            kernel_sizes (List[int]): Kernel size for each convolutional layer.
                - There should be one less kernel size than the number of channels.
            strides (Optional[Union[List[int], int, List[(int, int)]]): Stride for each convolutional layer.
                - If a single value, the same stride is used for all layers.
            paddings (Optional[Union[List[int], int, str, List[str], List[tuple], tuple]): Padding for each convolutional layer.
                - If a single value, the same padding is used for all layers.
            dilations (Union[List[int], int, List[tuple], tuple]): Dilation for each convolutional layer.
                - If a single value, the same dilation is used for all layers.
            padding_modes (Union[List[str], str]): Padding mode for each convolutional layer.
                - If a single value, the same padding mode is used for all layers.
            groups (Union[List[int], int]): Number of groups for each convolutional layer.
                - If a single value, the same number of groups is used for all layers.
            cnn_bias (Union[List[bool], bool]): Whether to use bias for each convolutional layer.
                - If a single value, the same bias is used for all layers.
            cnn_batch_norm (Optional[Union[List[bool], bool]]): Whether to use batch normalization for each convolutional layer.
                - If None, batch normalization is not used.
                - If a single value, the same batch normalization is used for all layers.
            cnn_act_funcs (Optional[Union[List[nn.Module], nn.Module]]): Activation functions for each convolutional layer.
                - If None, no activation function is used.
                - If a single value, the same activation function is used for all layers.
            pooling (Optional[Union[List[str], str, nn.Module, List[nn.Module]]]): Pooling type for each pooling layer.
                - If None, no pooling is used.
                - If a single value, the same pooling type is used for all layers.
            pooling_kernel_sizes (Union[List[int], int]): Kernel size for each pooling layer.
                - If a single value, the same kernel size is used for all layers.
            pooling_strides (Union[List[int], int]): Stride for each pooling layer.
                - If a single value, the same stride is used for all layers.
            pooling_paddings (Union[List[int], int]): Padding for each pooling layer.
                - If a single value, the same padding is used for all layers.
            pooling_dilations (Union[List[int], int]): Dilation for each pooling layer.
                - If a single value, the same dilation is used for all layers.
            ceil_mode (Union[List[bool], bool]): Whether to use ceil mode for each pooling layer.
                - If a single value, the same ceil mode is used for all layers.
            conv_dim (int): Convolutional dimension. 1 for 1D, 2 for 2D, and 3 for 3D.
            flatten (bool): Whether to flatten the output of the final convolutional layer.

            FC (Optional[MLP]): Fully connected layer to be added after the convolutional layers.
                - If provided, any arguments related to the fully connected layer will be ignored.
            node_counts (Optional[List[int]]): Number of nodes in each fully connected layer.
            dropout_rates (Optional[Union[List[float], float]]): Dropout rates for each fully connected layer.
                - If None, dropout is not used.
                - If a single value, the same dropout rate is used for all layers.
            FC_batch_norm (bool): Whether to use batch normalization for the fully connected layers.
            FC_act_funcs (Optional[List[nn.Module]]): Activation functions for each fully connected layer.
                - If None, ReLU is used for all layers except the last layer.
                - If a single value, the same activation function is used for all layers except the last layer.
                - List can be equal to the number of layers or one less than the number of layers.
            # FC_create_input_count (bool): Whether to create the input count for the fully connected layer from the output convolutional layer.
            #     - If False, the input count must be provided in node_counts.
            #     - If True, calculated input count will be added to the beginning of node_counts.

            validation_method (ValidationMethod): Validation method
            loss_func (LossFunction): Loss function
        """
        super().__init__(validation_method, loss_func)

        # Convolutional layers
        if len(channels) != len(kernel_sizes) + 1:
            raise ValueError("Number of channels must be equal to the number of kernel sizes plus one.")

        strides = self._param_check(strides, [int, tuple], len(kernel_sizes))
        paddings = self._param_check(paddings, [int, str, tuple], len(kernel_sizes))
        dilations = self._param_check(dilations, [int, tuple], len(kernel_sizes))
        padding_modes = self._param_check(padding_modes, [str], len(kernel_sizes))
        groups = self._param_check(groups, [int], len(kernel_sizes))
        cnn_bias = self._param_check(cnn_bias, [bool], len(kernel_sizes))
        cnn_batch_norm = self._param_check(cnn_batch_norm, [bool], len(kernel_sizes), optional=True)
        cnn_act_funcs = self._param_check(cnn_act_funcs, [nn.Module], len(kernel_sizes), optional=True)
        pooling = self._param_check(pooling, [str, nn.Module], len(kernel_sizes), optional=True)
        pooling_kernel_sizes = self._pooling_param_check(pooling_kernel_sizes, [int], pooling)
        pooling_strides = self._pooling_param_check(pooling_strides, [int], pooling)
        pooling_paddings = self._pooling_param_check(pooling_paddings, [int], pooling)
        pooling_dilations = self._pooling_param_check(pooling_dilations, [int], pooling)
        ceil_modes = self._pooling_param_check(ceil_modes, [bool], pooling)

        if conv_dim == 1:
            Conv = nn.Conv1d
            BatchNorm = nn.BatchNorm1d
            MaxPool = nn.MaxPool1d
            AvgPool = nn.AvgPool1d
        elif conv_dim == 2:
            Conv = nn.Conv2d
            BatchNorm = nn.BatchNorm2d
            MaxPool = nn.MaxPool2d
            AvgPool = nn.AvgPool2d
        elif conv_dim == 3:
            Conv = nn.Conv3d
            BatchNorm = nn.BatchNorm3d
            MaxPool = nn.MaxPool3d
            AvgPool = nn.AvgPool3d
        else:
            raise ValueError("conv_dim must be 1, 2, or 3.")
        
        self.layers = []
        for indx, channel in enumerate(channels[:-1]):
            self.layers.append(Conv(channel, channels[indx + 1], kernel_sizes[indx], strides[indx], paddings[indx], padding_modes[indx], dilations[indx], groups[indx], cnn_bias[indx]))
            if cnn_batch_norm[indx]:
                self.layers.append(BatchNorm(channels[indx + 1]))
            if cnn_act_funcs[indx] is not None:
                self.layers.append(cnn_act_funcs[indx])
            if pooling[indx] is not None:
                if isinstance(pooling[indx], str):
                    if pooling[indx] == "max":
                        self.layers.append(MaxPool(pooling_kernel_sizes[indx], pooling_strides[indx], pooling_paddings[indx], pooling_dilations[indx], ceil_mode=ceil_modes[indx]))
                    elif pooling[indx] == "avg":
                        self.layers.append(AvgPool(pooling_kernel_sizes[indx], pooling_strides[indx], pooling_paddings[indx], pooling_dilations[indx], ceil_mode=ceil_modes[indx]))
                    else:
                        raise ValueError("Pooling type must be 'max' or 'avg'.")
                elif isinstance(pooling[indx], nn.Module):
                    self.layers.append(pooling[indx])
                elif pooling[indx] is not None:
                    raise ValueError("Pooling type must be 'max' or 'avg' or none.")
                
        if flatten:
            self.layers.append(nn.Flatten())

        # Fully connected layers
        if FC is None:
            FC = MLP(node_counts, dropout_rates, FC_batch_norm, FC_act_funcs)
        self.layers.append(FC.model)

        # Create the model
        self.model = nn.Sequential(*self.layers)


    def _param_check(self, param:any, types:list, length:int, optional:bool=False) -> list:
        """
        Check if the parameter is of the correct type and length.

        Args:
            param (any): Parameter to be checked
            types (list): List of types
            length (int): Length of the parameter

        Returns:
            list: Parameter
        """
        if optional and param is None:
            return [None] * length

        if isinstance(param, types):
            param = [param] * length
        elif isinstance(param, list):
            if len(param) == 1:
                param = param * length
            elif len(param) != length:
                raise ValueError(f"{param} must be equal to the number of kernel sizes or 1 for all layers.")
        else:
            raise ValueError(f"{param} must be an integer, tuple, or list.")
        return param
    
    def _pooling_param_check(self, param:any, types:list, pooling:list) -> list:
        """
        Check if the pooling parameter is of the correct type and length. This may vary if pooling is included as some layers may not have pooling.

        Args:
            param (any): Parameter to be checked
            types (list): List of types
            pooling (list): List of pooling types

        Returns:
            list: Parameter
        """
        if isinstance(param, types):
            param = [param] * len(pooling)
        elif not isinstance(param, list):
            raise ValueError(f"{param} must be an integer, tuple, or list.")
        elif len(param) == 1:
            param = param * len(pooling)
                
        if len(param) == len(pooling):
            for indx, pool in enumerate(pooling):
                if isinstance(pool, nn.Module):
                    continue
                if pool is not None and param[indx] is None:
                    raise ValueError("Pooling parameters must be provided for all pooling layers.")
        elif len(param) < len(pooling):
            indx = 0
            for pool in pooling:
                if isinstance(pool, nn.Module):
                    raise ValueError("Confusing problem not resolved.")
                if pool is not None:
                    if indx >= len(param) or param[indx] is None:
                        raise ValueError("Pooling parameters must be provided for all pooling layers.")
                    indx += 1
        else:
            raise ValueError("Too many pooling parameters provided.")
                    
        return param