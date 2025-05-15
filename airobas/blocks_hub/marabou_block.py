import logging
import time
from time import perf_counter
from typing import Dict

import numpy as np
from airobas.verif_pipeline import (
    BlockVerif,
    BlockVerifOutput,
    DataContainer,
    ProblemContainer,
    StatusVerif,
)
from keras.layers import Activation, Dense
from keras.models import Sequential, clone_model
from maraboupy import Marabou, MarabouCore
from maraboupy.MarabouNetwork import MarabouNetwork  # (pip install maraboupy)

logger = logging.getLogger(__name__)

output_name = "OUTPUT"


def separate_activations(model: Sequential):
    """
    Returns a new Keras Sequential model where Dense layer activations are
    separated into distinct Activation layers.

    This transformation is necessary for tools like Marabou, which expect
    activation functions (e.g., ReLU) to be explicitly represented in separate layers.

    Args:
        model: Original Keras Sequential model.

    Returns:
        Sequential: A new model with separate Activation layers.
    """
    new_model = Sequential()
    # copy_model = clone_model(model)
    for layer in model.layers:
        if isinstance(layer, Dense):
            # Add a new Dense layer without activation
            new_dense_layer = Dense(
                units=layer.units,
                input_shape=layer.input_shape[1:],
                kernel_initializer=layer.kernel_initializer,
                bias_initializer=layer.bias_initializer,
            )
            new_model.add(new_dense_layer)
            # Copy the weights from the original layer to the new layer
            new_weights = [layer.get_weights()[0]]  # Kernel weights
            new_biases = [layer.get_weights()[1]]  # Bias values
            new_dense_layer.set_weights(new_weights + new_biases)
            # Add the activation layer separately
            activation_name = layer.activation.__name__
            if activation_name != "linear":
                new_model.add(Activation(activation_name))
        else:
            # Add non-Dense layers as they are
            new_model.add(layer)
    return new_model


class MarabouSequential(MarabouNetwork):
    """
    A wrapper to convert a Keras Sequential model into a Marabou verifiable network.

    This class separates activation functions from dense layers (required by Marabou),
    initializes Marabou variables for each layer, builds the equation system, and enables
    verification queries via Marabou.

    Parameters
    ----------
    model : The Keras Sequential model to be parsed and verified.


    Attributes
    ----------
    keras_model : A modified version of the input model with separate activation layers.
    layers : List of layers in the modified Keras model.
    varMap : Maps each layer's name to its associated Marabou variables.
    inputVars : Marabou input variable indices.
    outputVars : Marabou output variable indices.
    """

    def __init__(self, model: Sequential):
        super().__init__()
        if not isinstance(model, Sequential):
            raise ValueError("model should be Sequential but is a {} object".format(type(model)))
        self.keras_model = separate_activations(model)
        self.layers = self.keras_model.layers
        self.varMap: Dict[str, int] = dict()
        self.build()

    def build(self):
        """
        Initializes and maps Marabou variables for each layer and constructs
        equations for Dense and ReLU layers.
        """
        for layer in self.layers:
            # get input_shape
            input_dim = np.prod(layer.input_shape[1:])
            self.varMap[layer.name] = []
            for i in range(input_dim):
                j = self.getNewVariable()
                self.varMap[layer.name].append(j)
        output_dim = np.prod(self.keras_model.output_shape[1:])
        global output_name
        self.varMap[output_name] = []
        for i in range(output_dim):
            j = self.getNewVariable()
            self.varMap[output_name].append(j)

        self._init_input_vars()
        self._init_output_vars()

        for i in range(len(self.layers)):
            self.buildEquations(i)

    def _init_input_vars(self):
        """
        Initializes the input variables for the Marabou network.
        """
        self.inputVars = np.asarray(self.varMap[self.layers[0].name], dtype="int")[None, None]

    def _init_output_vars(self):
        """
        Initializes the output variables for the Marabou network.
        """
        global output_name
        self.outputVars = np.asarray(self.varMap[output_name], dtype="int")[None, None]

    def buildEquations(self, index_layer, update_relu=True):
        """
        Constructs Marabou equations for a given layer.

        Parameters
        ----------
        index_layer : int
            Index of the layer to process.
        update_relu : bool, optional
            Whether to process ReLU activation layers (default is True).

        Raises
        ------
        NotImplementedError
            If the layer type is not supported (non-Dense, non-ReLU).
        """
        layer = self.layers[index_layer]

        if isinstance(layer, Dense):
            # do something
            self.add_dense(index_layer)
        elif isinstance(layer, Activation) and layer.get_config()["activation"] == "relu":
            # do something
            if update_relu:
                self.add_relu(index_layer)
        else:
            raise NotImplemented(layer)

    def get_output_layer(self, index_layer):
        """
        Returns the output variable mapping for a given layer.

        Parameters
        ----------
        index_layer : int
            The index of the current layer.

        Returns
        -------
        list
            A list of Marabou variable indices for the output.
        """

        global output_name
        if index_layer + 1 == len(self.layers):
            output_var = self.varMap[output_name]
        else:
            output_var = self.varMap[self.layers[index_layer + 1].name]

        return output_var

    def add_dense(self, index_layer):
        """
        Adds equality constraints corresponding to a Dense layer.

        Parameters
        ----------
        index_layer : Index of the Dense layer in the model.
        """
        layer = self.layers[index_layer]
        input_var = self.varMap[layer.name]
        output_var = self.get_output_layer(index_layer)

        kernel = layer.kernel
        if layer.use_bias:
            bias = list(layer.bias)
        else:
            bias = [0.0] * len(output_var)

        for j in range(kernel.shape[1]):
            vars_eq = [output_var[j]]
            coeffs_eq = [-1.0]
            vars_eq += input_var
            coeffs_eq += list(kernel[:, j])
            scalar = -bias[j]
            self.addEquality(vars_eq, coeffs_eq, scalar, isProperty=False)

    def add_relu(self, index_layer):
        """
        Adds ReLU constraints to Marabou for an activation layer.

        Parameters
        ----------
        index_layer : Index of the ReLU activation layer.
        """
        print("relu")
        layer = self.layers[index_layer]
        input_var = self.varMap[layer.name]
        output_var = self.get_output_layer(index_layer)

        if len(input_var) != len(output_var):
            raise ValueError("input and output dimension of ReLU layer do not match")

        for i, j in zip(input_var, output_var):
            self.addRelu(i, j)

    def get_output_dim(self):
        global output_name
        return len(self.varMap[output_name])

    def solve_query(self, options=None):
        if options is None:
            result = self.solve(verbose=False)
        else:
            result = self.solve(verbose=False, options=options)
        input_sat = None
        output_sat = None

        if result[0] == "sat":
            n_in = len(self.inputVars[0][0])
            n_out = len(self.outputVars[0][0])
            input_sat = np.array([result[1][self.inputVars[0][0][i]] for i in range(n_in)])
            output_sat = np.array([result[1][self.outputVars[0][0][i]] for i in range(n_out)])
        if result[0] == "TIMEOUT":
            logger.info(f"Time out !")
        return (
            [result[0] == "sat", result[0] == "unsat", result[0] == "TIMEOUT"],
            input_sat,
            output_sat,
        )


def solve_stability_property(network: MarabouSequential, x_min, x_max, y_min, y_max, timeout=0):
    t_init = time.perf_counter()
    # Set Lower and Upper bound for the input perturbation
    for i, x_min_i in enumerate(x_min):
        network.setLowerBound(network.inputVars[0][0][i], x_min_i)
    for i, x_max_i in enumerate(x_max):
        network.setUpperBound(network.inputVars[0][0][i], x_max_i)
    # find a sample that is either greater than Y_max or lower than Y_min
    equ_list = []

    for i in range(network.get_output_dim()):
        if np.isinf(y_min[i]) or np.isinf(y_max[i]):
            continue
        equ_l = MarabouCore.Equation(MarabouCore.Equation.LE)  # greater or equal >= scalar
        equ_l.addAddend(1, network.outputVars[0][0][i])
        equ_l.setScalar(y_min[i])
        # equ_l : f(x)[i]< Y_min[i]
        equ_u = MarabouCore.Equation(MarabouCore.Equation.GE)  # greater or equal >= scalar
        equ_u.addAddend(1, network.outputVars[0][0][i])
        equ_u.setScalar(y_max[i])
        # equ_u : f(x)[i]> Y_max[i]
        equ_list.append([equ_l])  # one disjunction
        equ_list.append([equ_u])  # one disjunction

    network.addDisjunctionConstraint(equ_list)
    t_end_init = time.perf_counter()
    options = None
    if timeout:
        options = Marabou.createOptions(timeoutInSeconds=int(timeout), verbosity=0)
    else:
        options = Marabou.createOptions(verbosity=0)
    result = network.solve_query(options=options)
    t_end_solve = time.perf_counter()
    network.clearProperty()
    network.disjunctionList = []
    return result, (t_init, t_end_init, t_end_solve)


class MarabouBlock(BlockVerif):
    def __init__(
        self,
        problem_container: ProblemContainer,
        data_container: DataContainer,
        **kwargs,
    ):
        super().__init__(problem_container=problem_container, data_container=data_container)
        self.options = kwargs

    def verif(self, indexes: np.ndarray) -> BlockVerifOutput:
        nb_points = len(indexes)
        output = BlockVerifOutput(
            status=np.array([StatusVerif.UNKNOWN for i in range(nb_points)], dtype=StatusVerif),
            inputs=[None for i in range(nb_points)],
            outputs=[None for i in range(nb_points)],
            build_time=0,
            init_time_per_sample=np.empty(nb_points, dtype=float),
            verif_time_per_sample=np.empty(nb_points, dtype=float),
        )
        t1 = perf_counter()
        network = MarabouSequential(model=self.problem_container.model)
        t2 = perf_counter()
        output.build_time = t2 - t1
        x_min = self.data_container.lbound_input_points[indexes, :]
        x_max = self.data_container.ubound_input_points[indexes, :]
        y_min = self.data_container.lbound_output_points[indexes, :]
        y_max = self.data_container.ubound_output_points[indexes, :]
        for index in range(nb_points):
            ((score, input_sat, output_sat), times) = solve_stability_property(
                network,
                x_min=x_min[index],
                x_max=x_max[index],
                y_min=y_min[index],
                y_max=y_max[index],
                timeout=self.options.get("time_out", 200),
            )
            output.init_time_per_sample[index] = times[1] - times[0]
            output.verif_time_per_sample[index] = times[2] - times[1]
            status = StatusVerif.UNKNOWN
            if score[0]:
                # Found counter example
                status = StatusVerif.VIOLATED
                output.inputs[index] = input_sat
                output.outputs[index] = output_sat
            if score[1]:
                status = StatusVerif.VERIFIED
            if score[2]:
                status = StatusVerif.TIMEOUT
            output.status[index] = status
            logger.info(f"Current Verified (%) {np.sum(output.status == StatusVerif.VERIFIED) / nb_points * 100}")
            logger.info(f"Current Violated (%) {np.sum(output.status == StatusVerif.VIOLATED) / nb_points * 100}")
            logger.info(f"Current Timeout (%) {np.sum(output.status == StatusVerif.TIMEOUT) / nb_points * 100}")
        return output

    @staticmethod
    def get_name() -> str:
        return "marabou-verif"
