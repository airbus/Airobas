import airobas.blocks_hub.mip_blocks_lib.commons.bounds_propagation.bounds_function_linear as bounds_linear
from airobas import blocks_hub as bounds_relu
from airobas.blocks_hub.mip_blocks_lib.commons.layers import InputLayer, Relu
from airobas.blocks_hub.mip_blocks_lib.commons.neural_network import NeuralNetwork
from airobas.blocks_hub.mip_blocks_lib.commons.parameters import Bounds


def compute_bounds_linear_propagation(
    lmodel: NeuralNetwork,
    runtime=False,
    binary_vars=None,
    start=-1,
    end=-1,
    method: Bounds = None,
):
    if method is None:
        method = Bounds.SYMBOLIC_INT_ARITHMETIC
    if start == -1:
        # begin computation from the input layer
        lmodel.input.compute_bounds()
        p_l = lmodel.input
    else:
        # begin computation from the specified hidden layer
        p_l = lmodel.layers[start]
    if end == -1:
        # end computation at the output layer
        end = len(lmodel.layers) - 1
    for l in range(start + 1, end + 1):
        # bounds for relu nodes
        if isinstance(lmodel.layers[l], InputLayer):
            continue
        if isinstance(lmodel.layers[l], Relu):
            if runtime:
                b_v = binary_vars[l]
            else:
                b_v = []
            bounds_relu.compute_bounds(
                current_layer=lmodel.layers[l],
                previous_layer=p_l,
                method=method,
                runtime=runtime,
                binary_vars=b_v,
                input_bounds=lmodel.input.bounds,
            )
        else:
            bounds_linear.compute_bounds(
                current_layer=lmodel.layers[l],
                previous_layer=p_l,
                method=method,
                input_bounds=lmodel.input.bounds,
            )
        p_l = lmodel.layers[l]
