try:
    from decomon.models.utils import ConvertMethod
except ImportError as e:
    print(e)
from blocks_hub.mip_blocks_lib.commons.bounds_propagation.bounds_computation_decomon import (
    BoundsComputationDecomon,
)
from blocks_hub.mip_blocks_lib.commons.bounds_propagation.bounds_computation_interface import (
    BoundComputation,
)
from blocks_hub.mip_blocks_lib.commons.bounds_propagation.bounds_computation_symbolic_arithmetic import (
    BoundComputationSymbolicArithmetic,
    default_propagation,
)
from blocks_hub.mip_blocks_lib.commons.parameters import (
    ParamsBoundComputation,
    ParamsBoundComputationEnum,
)


def create_object_bounds(params: ParamsBoundComputation, neural_network):
    if params.params_bound_computation_enum == ParamsBoundComputationEnum.SIA:
        return BoundComputationSymbolicArithmetic(neural_network)
    if params.params_bound_computation_enum == ParamsBoundComputationEnum.DECOMON:
        return BoundsComputationDecomon(
            neural_network, convert_method=ConvertMethod.CROWN
        )
    if params.params_bound_computation_enum == ParamsBoundComputationEnum.MILP:
        return BoundComputationSolverMilp(neural_network, **params.params)
