import logging
import time
from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Type, Union

import keras
import numpy as np

logger = logging.getLogger(__name__)


class StatusVerif(Enum):
    VERIFIED = 0
    VIOLATED = 1
    UNKNOWN = 2  # incomplete methods cannot conclude
    TIMEOUT = 3  # this is more for complete methods that runs out of time.


class BoundsDomainParameter:
    @abstractmethod
    def compute_lb_ub_bounds(self, x: np.ndarray) -> np.ndarray:
        ...


@dataclass
class StabilityProperty:
    input_bound_domain_param: BoundsDomainParameter
    output_bound_domain_param: BoundsDomainParameter


@dataclass
class BoundsDomainBoxParameter(BoundsDomainParameter):
    rel_noise: Optional[float] = None
    abs_noise: Optional[float] = None
    use_relative: bool = True

    def compute_lb_ub_bounds(self, x: np.ndarray) -> np.ndarray:
        if self.use_relative:
            return x - self.rel_noise * np.abs(x), x + self.rel_noise * np.abs(x)
        return x - self.abs_noise, x + self.abs_noise


@dataclass
class BoundsDomainBoxParameterPerValueInterval(BoundsDomainParameter):
    bounds_domain_parameters: List[Tuple[float, float, BoundsDomainParameter]]

    def compute_lb_ub_bounds(self, x: np.ndarray) -> np.ndarray:
        merge_lb, merge_ub = np.empty(x.shape), np.empty(x.shape)
        for val1, val2, bparameters in self.bounds_domain_parameters:
            indexes = np.nonzero(np.logical_and(val1 < x, x <= val2))
            lb, ub = bparameters.compute_lb_ub_bounds(x[indexes])
            merge_lb[indexes] = lb
            merge_ub[indexes] = ub
        return merge_lb, merge_ub


@dataclass
class ProblemContainer:
    def __init__(
        self,
        tag_id: Union[str, int],
        model: keras.Model,
        stability_property: StabilityProperty,
    ):
        self.tag_id = tag_id
        self.model = model
        self.stability_property = stability_property


@dataclass
class DataContainer:
    # Utility class object to store the data used for this experiments.
    def __init__(
        self,
        input_points: np.ndarray,
        lbound_input_points: np.ndarray,
        ubound_input_points: np.ndarray,
        output_points: np.ndarray,
        lbound_output_points: np.ndarray,
        ubound_output_points: np.ndarray,
    ):
        # self.tag_id = tag_id
        # self.model = model
        # self.stability_property = stability_property
        self.input_points = input_points
        self.lbound_input_points = lbound_input_points
        self.ubound_input_points = ubound_input_points
        self.output_points = output_points
        self.lbound_output_points = lbound_output_points
        self.ubound_output_points = ubound_output_points


def compute_bounds(
    stability_property: StabilityProperty, x: np.ndarray, is_input: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    if is_input:
        return stability_property.input_bound_domain_param.compute_lb_ub_bounds(x)
    else:
        return stability_property.output_bound_domain_param.compute_lb_ub_bounds(x)


@dataclass
class BlockVerifOutput:
    """
    What a bloc verif will return
    """

    status: np.ndarray  # List of status
    inputs: List[Optional[np.ndarray]]  # Counter example input points (or None)
    outputs: List[Optional[np.ndarray]]  # Counter example prediction (or None)
    build_time: float  # [first initialisation of the block]
    # compile/model construction time per sample before the actual verification queries
    init_time_per_sample: np.ndarray
    # actual verification time when model has been compiled
    verif_time_per_sample: np.ndarray


@dataclass
class GlobalVerifOutput:
    # List of methods names : size = size of block
    methods: List[str]
    #
    # solvers: List[]

    # detailed historic per bloc, and remember the original index point that enters in a given block of verif !
    results: List[Tuple[BlockVerifOutput, Optional[List[int]]]]
    # final status after running possibly several function
    status: np.ndarray  # shape : size of the dataset

    # for each index point tested, return the name of the verification method that concluded on this point.
    index_block_that_concluded: np.ndarray

    # Those could be recomputed from BlockVerifOutput
    inputs: List[Optional[np.ndarray]]  # Counter example input points (or None)
    outputs: List[Optional[np.ndarray]]  # Counter example prediction (or None)

    build_time: float  # sum of individual build time from BlockVerifOutputs(s).
    init_time_per_sample: np.ndarray  # sum of init_time_per_sample from BlockVerifOutputs
    verif_time_per_sample: np.ndarray  # sum of verif_time_per_sample from BlockVerifOutputs


class BlockVerif:
    def __init__(
        self, problem_container: ProblemContainer, data_container: DataContainer
    ):
        self.problem_container = problem_container
        self.data_container = data_container

    @abstractmethod
    def verif(self, indexes: np.ndarray) -> BlockVerifOutput:
        ...

    @staticmethod
    def get_name() -> str:
        ...


def full_verification_pipeline(
    problem: ProblemContainer,
    input_points: np.ndarray,
    # data points to test property on
    output_points: Optional[np.ndarray],
    # could be model inference of input_points, or ground truth etc.
    blocks_verifier: List[Union[Type[BlockVerif], Dict]],
    verbose: bool = True,
) -> GlobalVerifOutput:
    if output_points is None:
        output_points = problem.model.predict(input_points, verbose=0)
    t_0 = time.perf_counter()
    x_min, x_max = compute_bounds(
        problem.stability_property, input_points, is_input=True
    )
    # Bounds for input ..
    y_min, y_max = compute_bounds(
        problem.stability_property, output_points, is_input=False
    )
    # Bounds for desired output
    data = DataContainer(input_points, x_min, x_max, output_points, y_min, y_max)
    nb_points = input_points.shape[0]
    global_verif_output = GlobalVerifOutput(
        methods=[b[0].__name__ for b in blocks_verifier],
        results=[],
        status=np.array(
            [StatusVerif.UNKNOWN for _ in range(nb_points)], dtype=StatusVerif
        ),
        index_block_that_concluded=np.empty(nb_points, dtype=int),
        inputs=[None for _ in range(nb_points)],
        outputs=[None for _ in range(nb_points)],
        build_time=0,
        init_time_per_sample=np.empty(nb_points, dtype=float),
        verif_time_per_sample=np.empty(nb_points, dtype=float),
    )
    index = np.arange(nb_points)
    index_method = 0
    for method, kwargs in blocks_verifier:
        if not len(index):
            break
        t_start = time.perf_counter()
        verifier = method(problem_container=problem, data_container=data, **kwargs)
        logger.info(f"Running...{method.get_name()}, {len(index)} points to be tested")
        res: BlockVerifOutput = verifier.verif(indexes=index)
        t_end = time.perf_counter()
        global_verif_output.results.append((res, list(index)))
        for i in range(len(index)):
            global_verif_output.inputs[index[i]] = res.inputs[i]
            global_verif_output.outputs[index[i]] = res.outputs[i]
            if res.status[i] in {StatusVerif.VERIFIED, StatusVerif.VIOLATED}:
                global_verif_output.index_block_that_concluded[index[i]] = index_method
        global_verif_output.status[index] = res.status
        global_verif_output.init_time_per_sample[index] = res.init_time_per_sample
        global_verif_output.verif_time_per_sample[index] = res.verif_time_per_sample
        # update index with only the indices that are neither sat or unsat
        index = np.nonzero(
            np.logical_or(
                global_verif_output.status == StatusVerif.UNKNOWN,
                global_verif_output.status == StatusVerif.TIMEOUT,
            )
        )[0]
        logger.info(f"Remaining index {len(index)}")
        logger.info(
            f"Current verified (%) {np.sum(global_verif_output.status==StatusVerif.VERIFIED)/nb_points*100}"
        )
        logger.info(
            f"Current violated (%) {np.sum(global_verif_output.status==StatusVerif.VIOLATED)/nb_points*100}"
        )
        logger.info(f"{t_end-t_start} sec of computing for block {method.__name__}")
        index_method += 1
    # accuracy
    t_n = time.perf_counter()
    logger.info(f"Total time of verif : {t_n-t_0} seconds")
    return global_verif_output
