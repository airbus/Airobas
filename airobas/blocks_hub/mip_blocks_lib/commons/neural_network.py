import os
from dataclasses import dataclass
from functools import reduce
from typing import Dict, List, Optional, Set, Tuple, Union

import keras
import numpy as np
import onnx
import onnx.numpy_helper
from airobas.blocks_hub.mip_blocks_lib.commons.layers import (
    InputLayer,
    InputType,
    Layer,
    Linear,
    Relu,
)
from keras.activations import linear, relu
from keras.layers import Activation, Dense
from keras.models import Sequential


@dataclass
class InputBoundsNeuralNetwork:
    input_lower_bounds: np.ndarray
    input_upper_bounds: np.ndarray


class NeuralNetwork:
    def __init__(self):
        self.input: Optional[InputLayer] = None
        self.layers: List[Union[Layer, Relu]] = []
        self.all_keys: Set[Tuple] = set()
        self.all_keys_per_layer: Dict[int, List[Tuple[int, int]]] = {}
        self.nb_keys_relu = None
        self.nb_layers = 0

    def neurons_flatten(self):
        return reduce(
            lambda x, y: x + self.all_keys_per_layer[y],
            sequence=range(len(self.layers)),
        )

    def get_bounds(self, keys):
        l_pre = [self.get_lb(id_layer=k[0], node_in_layer=k[1], in_or_out="in") for k in keys]
        u_pre = [self.get_ub(id_layer=k[0], node_in_layer=k[1], in_or_out="in") for k in keys]
        l_post = [self.get_lb(id_layer=k[0], node_in_layer=k[1], in_or_out="out") for k in keys]
        u_post = [self.get_ub(id_layer=k[0], node_in_layer=k[1], in_or_out="out") for k in keys]
        return l_pre, u_pre, l_post, u_post

    def compute_keys(self):
        self.all_keys = [(i, x) for i in range(len(self.layers)) for x in range(self.layers[i].output_shape)]
        self.all_keys_per_layer = {
            i: [(i, x) for x in range(self.layers[i].output_shape)] for i in range(len(self.layers))
        }
        self.nb_keys_relu = sum(
            [len(self.all_keys_per_layer[i]) for i in self.all_keys_per_layer if isinstance(self.layers[i], Relu)]
        )
        self.nb_layers = len(self.layers)

    def get_lb(self, id_layer, node_in_layer, in_or_out: str):
        return self.layers[id_layer].bounds[in_or_out]["l"][node_in_layer]

    def get_ub(self, id_layer, node_in_layer, in_or_out: str):
        return self.layers[id_layer].bounds[in_or_out]["u"][node_in_layer]

    def get_keys_of_layer(self, i: int):
        return self.all_keys_per_layer[i]

    def load(self, path, spec):
        _, model_format = os.path.splitext(path)
        if model_format == ".h5":
            keras_model = keras.models.load_model(path, compile=False)
            self.parse_keras(keras_model, spec)
            return keras_model
        elif model_format == ".onnx":
            onnx_model = onnx.load(path)
            self.parse_onnx(onnx_model, spec)
        else:
            raise Exception("Unsupported model format")

    def parse_onnx(
        self,
        model,
        spec: InputBoundsNeuralNetwork,
        input_type: InputType = InputType.FLOAT,
        additional_layers: Optional[List[Layer]] = None,
    ):
        # input constraints
        self.input = InputLayer(
            input_shape=len(spec.input_lower_bounds),
            lower_bounds=spec.input_lower_bounds,
            upper_bounds=spec.input_upper_bounds,
            input_type=input_type,
        )
        i = 1
        nodes = model.graph.node
        self.layers = [self.input]
        if additional_layers is not None:
            self.layers += additional_layers
            i = len(self.layers)
        for j in range(len(nodes)):
            node = nodes[j]
            next_node = nodes[j + 1] if j < len(nodes) - 1 else nodes[j]
            if node.op_type in ["Flatten", "Relu"]:
                pass
            elif node.op_type == "Gemm":
                [weights] = [onnx.numpy_helper.to_array(t) for t in model.graph.initializer if t.name == node.input[1]]
                [bias] = [onnx.numpy_helper.to_array(t) for t in model.graph.initializer if t.name == node.input[2]]
                if next_node.op_type == "Relu":
                    self.layers.append(Relu(weights.shape[0], weights, bias, i))
                else:
                    self.layers.append(Linear(weights.shape[0], weights, bias, i))
                i += 1
            else:
                pass
        self.compute_keys()

    def parse_keras(self, model: keras.Model, spec: InputBoundsNeuralNetwork):
        # input constraints
        self.input = InputLayer(
            input_shape=len(spec.input_lower_bounds),
            lower_bounds=spec.input_lower_bounds,
            upper_bounds=spec.input_upper_bounds,
        )
        self.layers = [self.input]
        # layers of the network
        for i in range(len(model.layers)):
            lay = model.layers[i]
            if "activation" not in lay.__dict__.keys():
                continue
            if lay.activation == relu:
                self.layers.append(
                    Relu(
                        lay.output_shape[1],
                        lay.get_weights()[0].T,
                        lay.get_weights()[1],
                        i + 1,
                    )
                )
            elif lay.activation == linear:
                self.layers.append(
                    Linear(
                        lay.output_shape[1],
                        lay.get_weights()[0].T,
                        lay.get_weights()[1],
                        i + 1,
                    )
                )
        self.compute_keys()

    def get_nb_fixed(self):
        ids = [i for i in range(len(self.layers)) if isinstance(self.layers[i], Relu)]
        return len([x for i in ids for x in self.layers[i].get_fixed()])

    def get_active_ids(self):
        ids = [i for i in range(len(self.layers)) if isinstance(self.layers[i], Relu)]
        return [(i, x) for i in ids for x in self.layers[i].get_all_active()]

    def get_inactive_ids(self):
        ids = [i for i in range(len(self.layers)) if isinstance(self.layers[i], Relu)]
        return [(i, x) for i in ids for x in self.layers[i].get_all_inactive()]

    def get_unstable(self):
        ids = [i for i in range(len(self.layers)) if isinstance(self.layers[i], Relu)]
        return [(i, x) for i in ids for x in self.layers[i].get_not_fixed()]

    def clone(self, spec: Optional[InputBoundsNeuralNetwork] = None):
        new_model = NeuralNetwork()
        for layer in self.layers:
            new_model.layers.append(layer.clone())
        if spec is None:
            # new_model.output = self.output.clone()
            new_model.input = self.input.clone()
        else:
            new_model.input = InputLayer(
                input_shape=len(spec.input_lower_bounds),
                lower_bounds=spec.input_lower_bounds,
                upper_bounds=spec.input_upper_bounds,
            )
            new_model.layers[0] = new_model.input
        new_model.compute_keys()
        return new_model

    def clean_vars(self):
        for layer in self.layers + [self.input]:
            layer.clean_vars()


def extract_neural_network(neural_network: NeuralNetwork, id_layer_input: int, id_last_layer: int):
    # input constraints
    neural_net = NeuralNetwork()
    layer_input = neural_network.layers[id_layer_input]
    input = InputLayer(
        input_shape=layer_input.output_shape,
        lower_bounds=layer_input.bounds["out"]["l"],
        upper_bounds=layer_input.bounds["out"]["u"],
    )
    neural_net.input = input
    neural_net.layers = [input]
    # layers of the network
    for j in range(id_layer_input + 1, id_last_layer + 1):
        neural_net.layers.append(neural_network.layers[j].clone())
    neural_net.compute_keys()
    return neural_net


def neural_network_to_keras(neural_network: NeuralNetwork):
    input_dim = neural_network.input.output_shape
    layers = []
    weights = []
    for j in range(len(neural_network.layers) - 1):
        true_layer = neural_network.layers[j + 1]
        if j == 0:
            layers += [Dense(true_layer.output_shape, input_dim=input_dim)]
        else:
            if isinstance(true_layer, Linear):
                layers += [Dense(true_layer.output_shape, activation="linear")]
            else:
                layers += [Dense(true_layer.output_shape)]
        weights += [true_layer.weights.T, true_layer.bias]
        if isinstance(true_layer, Relu):
            layers += [Activation("relu")]
    model = Sequential(layers=layers)
    model.set_weights(weights)
    return model
