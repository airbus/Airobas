from dataclasses import dataclass
from enum import Enum
from typing import Optional


class EncType(Enum):
    IDEAL = 0
    MUL_CHOICE = 1
    BIG_M = 2


class Bounds(Enum):
    INT_ARITHMETIC = 0
    SYMBOLIC_INT_ARITHMETIC = 1
    CONST = 2


class ParamsBoundComputationEnum(Enum):
    SIA = 0
    DECOMON = 1


class ParamsBoundComputation:
    def __init__(
        self,
        params_bound_computation_enum: Optional[ParamsBoundComputationEnum] = None,
        **kwargs
    ):
        self.params_bound_computation_enum = params_bound_computation_enum
        self.params = kwargs

    @staticmethod
    def default():
        return ParamsBoundComputation(
            params_bound_computation_enum=ParamsBoundComputationEnum.SIA
        )
