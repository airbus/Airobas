from abc import abstractmethod

from airobas.blocks_hub.mip_blocks_lib.commons.bounds_propagation.linear_propagation import (
    compute_bounds_linear_propagation,
)
from airobas.blocks_hub.mip_blocks_lib.commons.neural_network import NeuralNetwork
from airobas.blocks_hub.mip_blocks_lib.commons.parameters import Bounds

def default_propagation(neural_network: NeuralNetwork):
    compute_bounds_linear_propagation(
        lmodel=neural_network,
        runtime=False,
        binary_vars=None,
        start=-1,
        end=-1,
        method=Bounds.SYMBOLIC_INT_ARITHMETIC,
    )


class BoundComputation:
    def __init__(self, neural_network: NeuralNetwork):
        self.neural_network = neural_network

    @abstractmethod
    def update_bounds(self):
        ...

    def update_bounds_net(self, neural_network: NeuralNetwork):
        self.neural_network = neural_network
        self.update_bounds()
