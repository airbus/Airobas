from blocks_hub.mip_blocks_lib.commons.bounds_propagation.bounds_computation_interface import (
    BoundComputation,
)
from blocks_hub.mip_blocks_lib.commons.bounds_propagation.linear_propagation import (
    compute_bounds_linear_propagation,
)
from blocks_hub.mip_blocks_lib.commons.neural_network import NeuralNetwork
from blocks_hub.mip_blocks_lib.commons.parameters import Bounds


def default_propagation(neural_network: NeuralNetwork):
    compute_bounds_linear_propagation(
        lmodel=neural_network,
        runtime=False,
        binary_vars=None,
        start=-1,
        end=-1,
        method=Bounds.SYMBOLIC_INT_ARITHMETIC,
    )


class BoundComputationSymbolicArithmetic(BoundComputation):
    def __init__(
        self,
        neural_network: NeuralNetwork,
        bounds_method: Bounds = Bounds.SYMBOLIC_INT_ARITHMETIC,
    ):
        super().__init__(neural_network)
        self.bounds_method = bounds_method

    def update_bounds(self):
        compute_bounds_linear_propagation(
            lmodel=self.neural_network,
            runtime=False,
            binary_vars=None,
            start=-1,
            end=-1,
            method=self.bounds_method,
        )
