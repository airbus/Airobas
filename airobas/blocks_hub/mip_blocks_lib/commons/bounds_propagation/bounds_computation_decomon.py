from typing import Optional

# import decomon
import numpy as np

try:
    from decomon import get_lower_box, get_upper_box
    from decomon.backward_layers.backward_layers import BackwardActivation

    # from decomon.models import clone
    from decomon.models import clone
    from decomon.models.utils import ConvertMethod
except ImportError as e:
    print(e)
import logging

from airobas.blocks_hub.mip_blocks_lib.commons.bounds_propagation.bounds_computation_interface import (
    BoundComputation,
)
from airobas.blocks_hub.mip_blocks_lib.commons.bounds_propagation.linear_propagation import (
    Bounds,
    compute_bounds_linear_propagation,
)
from airobas.blocks_hub.mip_blocks_lib.commons.layers import Linear, Relu
from airobas.blocks_hub.mip_blocks_lib.commons.neural_network import (
    NeuralNetwork,
    neural_network_to_keras,
)
from tensorflow.python.keras.models import Model  # , Sequential, load_model

logger = logging.getLogger(__file__)


class BoundsComputationDecomon(BoundComputation):
    def __init__(
        self,
        neural_network: NeuralNetwork,
        convert_method: Optional["ConvertMethod"] = None,
        keras_network: Optional[Model] = None,
        compute_intermediary: Optional[bool] = False,
        save_to_debug: bool = False,
    ):
        if convert_method is None:
            convert_method = ConvertMethod.CROWN
        assert convert_method == ConvertMethod.CROWN
        super().__init__(neural_network)
        if keras_network is None:
            self.keras_network = neural_network_to_keras(self.neural_network)
        else:
            self.keras_network = keras_network
        self.compute_intermediary = compute_intermediary
        self.convert_method = convert_method
        self.save_to_debug = save_to_debug
        self.previous_value = []
        self.new_value = []
        self.decomon_model = clone(self.keras_network, method=self.convert_method)

    def update_bounds(self):
        compute_bounds_linear_propagation(
            lmodel=self.neural_network,
            runtime=False,
            binary_vars=None,
            start=-1,
            end=-1,
            method=Bounds.SYMBOLIC_INT_ARITHMETIC,
        )
        x_min = np.reshape(
            self.neural_network.input.bounds["in"]["l"],
            (1, 1, self.neural_network.input.bounds["in"]["l"].shape[0]),
        )
        x_max = np.reshape(
            self.neural_network.input.bounds["in"]["u"],
            (1, 1, self.neural_network.input.bounds["in"]["u"].shape[0]),
        )
        prev_value = []
        new_value = []
        nb_fixed_before = self.neural_network.get_nb_fixed()
        if self.compute_intermediary:
            layer_relu = [layer for layer in self.decomon_model.layers if isinstance(layer, BackwardActivation)]

            def get_bound_relu(layer_relu, relus_bound):
                def func(x):
                    bounds = relus_bound.predict(x, verbose=0)
                    return dict([(i, u_i) for (i, u_i) in zip(range(len(layer_relu)), bounds)])

                return func

            relus_bounds_upper = Model(
                self.decomon_model.inputs,
                [layer_i.get_input_at(0)[0] for layer_i in layer_relu],
            )
            relus_bounds_lower = Model(
                self.decomon_model.inputs,
                [layer_i.get_input_at(0)[1] for layer_i in layer_relu],
            )
            get_upper_relu = get_bound_relu(layer_relu, relus_bounds_upper)
            get_lower_relu = get_bound_relu(layer_relu, relus_bounds_lower)

            values = np.concatenate(
                [
                    x_min,
                    x_max,
                ],
                1,
            )
            value_up = get_upper_relu(values)
            value_down = get_lower_relu(values)
            layers_of_interest = [
                j for j in range(self.neural_network.nb_layers) if isinstance(self.neural_network.layers[j], Relu)
            ]

            for i in value_down:
                corresponding_layer = self.neural_network.layers[layers_of_interest[i]]
                if self.save_to_debug:
                    prev_value.append({})
                    new_value.append({})
                logger.debug(
                    f"norm diff :{np.linalg.norm(corresponding_layer.bounds['in']['l'] - value_down[i][0, :])}"
                )
                if self.save_to_debug:
                    prev_value[-1]["in_l"] = np.array(corresponding_layer.bounds["in"]["l"])
                    prev_value[-1]["out_l"] = np.array(corresponding_layer.bounds["out"]["l"])
                    prev_value[-1]["in_u"] = np.array(corresponding_layer.bounds["in"]["u"])
                    prev_value[-1]["out_u"] = np.array(corresponding_layer.bounds["out"]["u"])

                corresponding_layer.bounds["in"]["l"] = value_down[i][0, :]

                logger.debug(f"Value down shape : {value_down[i].shape}")
                if isinstance(corresponding_layer, Relu):
                    corresponding_layer.bounds["out"]["l"] = np.maximum(0, corresponding_layer.bounds["in"]["l"])
                else:
                    corresponding_layer.bounds["out"]["l"] = corresponding_layer.bounds["in"]["l"]
                logger.debug(f"Value up shape : {value_up[i].shape}")
                logger.debug(f"norm diff : {np.linalg.norm(corresponding_layer.bounds['in']['u'] - value_up[i][0, :])}")
                corresponding_layer.bounds["in"]["u"] = value_up[i][0, :]
                if isinstance(corresponding_layer, Relu):
                    corresponding_layer.bounds["out"]["u"] = np.maximum(0, corresponding_layer.bounds["in"]["u"])
                else:
                    corresponding_layer.bounds["out"]["u"] = corresponding_layer.bounds["in"]["u"]
                if self.save_to_debug:
                    new_value[-1]["in_l"] = value_down[i][0, :]
                    new_value[-1]["out_l"] = np.array(corresponding_layer.bounds["out"]["l"])
                    new_value[-1]["in_u"] = value_up[i][0, :]
                    new_value[-1]["out_u"] = np.array(corresponding_layer.bounds["out"]["u"])
        nb_fixed_after = self.neural_network.get_nb_fixed()
        # box = np.concatenate([X_min[:, None], X_max[:, None]], 1)
        value_up, value_down = self.decomon_model.predict(np.concatenate([x_min, x_max], axis=1))
        # value_up = get_upper_box(self.decomon_model, x_min, x_max)
        # value_down = get_lower_box(self.decomon_model, x_min, x_min)
        logger.debug(f"shape, value_up {value_up.shape}")
        logger.debug(f"shape, value_down {value_down.shape}")
        logger.debug(f"{value_down}")
        logger.debug(f"{value_up}")
        if self.save_to_debug:
            prev_value.append({})
            new_value.append({})
            prev_value[-1]["out_l"] = np.array(self.neural_network.layers[-1].bounds["out"]["l"])
            prev_value[-1]["out_u"] = np.array(self.neural_network.layers[-1].bounds["out"]["u"])
        self.neural_network.layers[-1].bounds["out"]["l"] = value_down[0, :]
        self.neural_network.layers[-1].bounds["out"]["u"] = value_up[0, :]
        if isinstance(self.neural_network.layers[-1], Linear):
            self.neural_network.layers[-1].bounds["in"]["l"] = np.copy(
                self.neural_network.layers[-1].bounds["out"]["l"]
            )
            self.neural_network.layers[-1].bounds["in"]["u"] = np.copy(
                self.neural_network.layers[-1].bounds["out"]["u"]
            )
        if self.save_to_debug:
            new_value[-1]["out_l"] = value_down[0, :]
            new_value[-1]["out_u"] = value_up[0, :]
        if self.save_to_debug:
            return nb_fixed_after, nb_fixed_before, prev_value, new_value
        return nb_fixed_after, nb_fixed_before

    # def update_bounds_net(self, neural_network: NeuralNetwork):
    #    self.neural_network = neural_network
    #    self.update_bounds()


def get_bound_relu(layer_relu, relus_bound):
    def func(x):
        bounds = relus_bound.predict(x)
        return dict([(l_i.name.split("_backward")[0], u_i) for (l_i, u_i) in zip(layer_relu, bounds)])

    return func
