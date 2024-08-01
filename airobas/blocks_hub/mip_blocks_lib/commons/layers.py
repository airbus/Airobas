# Mostly updated from https://github.com/vas-group-imperial/venus/blob/main/src/layers.py
from copy import deepcopy
from enum import Enum
from typing import Dict, Optional

import numpy as np

from airobas.blocks_hub.mip_blocks_lib.commons.linearfunctions import LinearFunctions


class LayerType(Enum):
    RELU = 0
    LINEAR = 1


class ReluState(Enum):
    ACTIVE = 0
    INACTIVE = 1
    UNSTABLE = 2
    NOT_COMPUTED = 3
    IRRELEVANT = 4


class InputType(Enum):
    FLOAT = 0
    INTEGER = 1


class Layer:
    def __init__(
        self,
        output_shape: int,
        weights: Optional[np.ndarray],
        bias: Optional[np.ndarray],
        depth: int,
        weights_minus: Optional[np.ndarray] = None,
        weights_plus: Optional[np.ndarray] = None,
    ):
        self.bounds: Dict[str, Dict[str, Optional[np.ndarray]]] = {
            "in": {"l": None, "u": None},
            "out": {"l": None, "u": None},
        }
        self.rnt_bounds: Dict[str, Dict[str, Optional[np.ndarray]]] = {
            "in": {"l": None, "u": None},
            "out": {"l": None, "u": None},
        }
        self.bound_equations: Dict[str, Dict[str, Optional[LinearFunctions]]] = {
            "in": {"l": None, "u": None},
            "out": {"l": None, "u": None},
        }
        self.rnt_bound_equations: Dict[str, Dict[str, Optional[LinearFunctions]]] = {
            "in": {"l": None, "u": None},
            "out": {"l": None, "u": None},
        }
        self.error: Dict[str, Optional[np.ndarray]] = {"out": None}
        self.output_shape = output_shape
        self.weights = weights
        self.weights_minus = weights_minus
        if weights_minus is None and weights is not None:
            self.weights_minus = np.minimum(self.weights, np.zeros(self.weights.shape))
        self.weights_plus = weights_plus
        if weights_plus is None and weights is not None:
            self.weights_plus = np.maximum(self.weights, np.zeros(self.weights.shape))
        self.bias = bias
        self.depth = depth

    def compute_bounds(self, **kwargs):
        ...

    def get_prime_lbound(self, node, p_layer, p_node):
        lb = p_layer.bounds["out"]["l"][p_node]
        if self.weights[node][p_node] < 0:
            lb = p_layer.bounds["out"]["u"][p_node]
        return lb

    def get_prime_ubound(self, node, p_layer, p_node):
        ub = p_layer.bounds["out"]["u"][p_node]
        if self.weights[node][p_node] < 0:
            ub = p_layer.bounds["out"]["l"][p_node]
        return ub

    def get_active_weights(self, node):
        w = np.nonzero(self.weights[node, :])[0]
        return w

    def clone(self):
        a = Layer(
            output_shape=self.output_shape,
            weights=self.weights,
            bias=self.bias,
            depth=self.depth,
            weights_plus=self.weights_plus,
            weights_minus=self.weights_minus,
        )
        return a

    def is_fixed(self, node):
        return False


class InputLayer(Layer):
    def __init__(
        self,
        input_shape: int,
        lower_bounds: Optional[np.ndarray],
        upper_bounds: Optional[np.ndarray],
        input_type: InputType = InputType.FLOAT,
    ):
        super().__init__(input_shape, None, None, 0)
        self.bounds: Dict[str, Dict[str, Optional[np.ndarray]]] = {
            "in": {"l": lower_bounds, "u": upper_bounds},
            "out": {"l": lower_bounds, "u": upper_bounds},
        }
        self.input_type = input_type
        self.compute_bounds()

    def compute_bounds(self):
        self.error = {"out": [0 for _ in range(self.output_shape)]}
        size = self.output_shape
        self.bound_equations["out"]["l"] = LinearFunctions(
            np.identity(size), np.zeros(size)
        )
        self.bound_equations["out"]["u"] = LinearFunctions(
            np.identity(size), np.zeros(size)
        )
        self.rnt_bounds = self.bounds
        self.rnt_bound_equations = self.bound_equations

    def clone(self):
        return InputLayer(
            self.output_shape,
            lower_bounds=deepcopy(self.bounds["in"]["l"]),
            upper_bounds=deepcopy(self.bounds["out"]["l"]),
            input_type=self.input_type,
        )


class Relu(Layer):
    def __init__(
        self,
        output_shape: int,
        weights: np.ndarray,
        bias: np.ndarray,
        depth: int,
        weights_plus: Optional[np.ndarray] = None,
        weights_minus: Optional[np.ndarray] = None,
    ):
        super().__init__(
            output_shape=output_shape,
            weights=weights,
            bias=bias,
            depth=depth,
            weights_minus=weights_minus,
            weights_plus=weights_plus,
        )
        self.error = {"in": [], "out": []}

    def clone(self):
        a = Relu(
            output_shape=self.output_shape,
            weights=self.weights,
            bias=self.bias,
            depth=self.depth,
            weights_minus=self.weights_minus,
            weights_plus=self.weights_plus,
        )
        return a

    def is_active(self, node: int):
        if self.bounds["in"]["l"][node] >= 0:
            return True
        else:
            return False

    def is_inactive(self, node: int):
        if self.bounds["in"]["u"][node] <= 0:
            return True
        else:
            return False

    def is_fixed(self, node: int):
        if self.bounds["in"]["l"][node] < 0 < self.bounds["in"]["u"][node]:
            return False
        else:
            return True

    def get_all_active(self):
        return [i for i in range(self.output_shape) if self.is_active(i)]

    def get_all_inactive(self):
        return [i for i in range(self.output_shape) if self.is_inactive(i)]

    def get_not_fixed(self, binary_vars=np.empty(0)):
        """
        :return nodes that are not fixed to either the active or the
        inactive state.
        """
        if len(binary_vars) == 0:
            return [i for i in range(self.output_shape) if not self.is_fixed(i)]
        else:
            return [
                i
                for i in range(self.output_shape)
                if not self.is_fixed(i) and binary_vars[i] != 0 and binary_vars[i] != 1
            ]

    def get_fixed(self, binary_vars=None):
        """
        :return nodes that are fixed to either the active or the
        inactive state.
        """
        if binary_vars is None or len(binary_vars) == 0:
            return [i for i in range(self.output_shape) if self.is_fixed(i)]
        else:
            return [
                i
                for i in range(self.output_shape)
                if self.is_fixed(i) and binary_vars[i] != 0 and binary_vars[i] != 1
            ]


class Linear(Layer):
    def clone(self):
        a = Linear(
            output_shape=self.output_shape,
            weights=self.weights,
            bias=self.bias,
            depth=self.depth,
            weights_minus=self.weights_minus,
            weights_plus=self.weights_plus,
        )
        # for key in self.__dict__.keys():
        #     setattr(a, key, deepcopy(getattr(self, key)))
        return a


class NormalisationLayer(Linear):
    def __init__(
        self, output_shape: int, shift: np.ndarray, scale: np.ndarray, depth: int
    ):
        matrix = np.diag(1 / scale)
        bias = -shift / scale
        super().__init__(
            output_shape=output_shape,
            weights=matrix,
            bias=bias,
            depth=depth,
            weights_minus=None,
            weights_plus=None,
        )
