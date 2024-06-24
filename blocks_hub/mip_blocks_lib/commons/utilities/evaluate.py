import numpy as np

from blocks_hub.mip_blocks_lib.commons.neural_network import Linear, NeuralNetwork, Relu


def random_input(neural_net: NeuralNetwork):
    input_lower = neural_net.input.bounds["in"]["l"]
    input_upper = neural_net.input.bounds["in"]["u"]
    random = np.random.random(neural_net.input.output_shape)
    return input_lower + random * (input_upper - input_lower)


def random_input_with_noise(neural_net: NeuralNetwork, init_input, max_delta):
    input_lower = neural_net.input.bounds["in"]["l"]
    input_upper = neural_net.input.bounds["in"]["u"]
    random = np.random.random(neural_net.input.output_shape)
    delta = -max_delta + 2 * random * max_delta
    return np.minimum(input_upper, np.maximum(init_input + delta, input_lower))


def evaluate(
    input_array: np.array, neural_net: NeuralNetwork, index_min=None, index_max=None
):
    if index_min is None:
        index_min, _ = default_index(neural_net)
    if index_max is None or index_max == -1:
        _, index_max = default_index(neural_net)
    values = [(input_array, input_array)]
    for k in range(index_min, index_max + 1):
        lay = neural_net.layers[k]
        if isinstance(neural_net.layers[k], Linear):
            nv = lay.weights.dot(values[-1][1]) + lay.bias
            values += [(nv, nv)]
        if isinstance(neural_net.layers[k], Relu):
            nv = lay.weights.dot(values[-1][1]) + lay.bias
            nvr = np.maximum(nv, 0)
            values += [(nv, nvr)]
    return values


def default_index(neural_net: NeuralNetwork):
    return 0, len(neural_net.layers) - 1


def get_relu_state(
    input_array: np.array,
    neural_net: NeuralNetwork,
    numerical_tolerance: float = 1e-7,
    index_min=None,
    index_max=None,
):
    if index_min is None:
        index_min, _ = default_index(neural_net)
    if index_max is None or index_max == -1:
        _, index_max = default_index(neural_net)
    evaluation = evaluate(
        input_array, neural_net, index_min=index_min, index_max=index_max
    )
    dict_relu = {}
    for j in range(len(evaluation)):
        lay = neural_net.layers[j]
        if isinstance(lay, Relu):
            for k in range(evaluation[j][0].shape[0]):
                if evaluation[j][0][k] < -numerical_tolerance:
                    dict_relu[(j, k)] = 0
                elif evaluation[j][0][k] > numerical_tolerance:
                    dict_relu[(j, k)] = 1
                else:
                    dict_relu[(j, k)] = -1
    return dict_relu
