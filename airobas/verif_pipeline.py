import logging
import time
from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Type, Union
#from .utils import merge_global_verif_outputs
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
    runtime_per_block: Dict[str,float] # sum of runtime of verification per method/block

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
    batch_split: int=1) -> GlobalVerifOutput:

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
    #loop for batches
    nb_samples=x_min.shape[0]#here we get nb_samples, with one sample it's always one?
    batch_size=nb_samples//batch_split+1
    
    if nb_samples % batch_split != 0:  # Handle uneven division
        batch_size += 1
    list_global_verif=[]
    t_n=0
    runtime_summary = {} # key: method.__name__, value: total runtime in sec 
    for i in range(batch_split):
        if verbose:
            print(f"Batch number {i}")
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, nb_samples)  # Ensure end_idx does not exceed array length
    
        input_sample_i = input_points[start_idx:end_idx]
        output_sample_i = output_points[start_idx:end_idx]
        
        x_min_i=x_min[start_idx:end_idx]
        x_max_i=x_max[start_idx:end_idx]

        y_min_i=y_min[start_idx:end_idx]
        y_max_i=y_max[start_idx:end_idx]

        assert (len(input_sample_i)==len(output_sample_i)),\
            f"Mismatch between input and output batch sizes: {len(input_sample_i)} vs {len(output_sample_i)}"

        if len(input_sample_i)==0:
            print(f"Batch {i + 1} is empty. Breaking.")
            break
        if verbose:
            print(f"Processing batch {i + 1}/{batch_split}:")
            print(f"Start index: {start_idx}, End index: {end_idx}")
            print(f"Input batch shape: {input_sample_i.shape}")
            print(f"Output batch shape: {output_sample_i.shape}")
        
        global_verif_output_i,t_n_i=full_verification_pipeline_batch(0,input_sample_i, x_min_i, x_max_i, 
                                        output_sample_i, y_min_i, y_max_i,blocks_verifier,problem)
        t_n+=t_n_i
        list_global_verif.append(global_verif_output_i)
    global_verif_output: GlobalVerifOutput=merge_global_verif_outputs(list_global_verif)
    logger.info(f"Total time of verif : {t_n-t_0} seconds")
    return global_verif_output

def full_verification_pipeline_batch(index_batch,input_points, x_min, x_max, output_points,\
                                    y_min, y_max,blocks_verifier,problem):
    # Bounds for desired output
    data = DataContainer(input_points, x_min, x_max, output_points, y_min, y_max)
    nb_points = x_min.shape[0]
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
        runtime_per_block={}
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
        global_verif_output.runtime_per_block[method.__name__]=t_end-t_start # key: method.__name__, value: runtime in sec
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
        logger.info(f"Treating batch number {index_batch}")
        logger.info(f"Remaining index {len(index)}")
        logger.info(
            f"Current verified (%) {np.sum(global_verif_output.status==StatusVerif.VERIFIED)/nb_points*100}"
        )
        logger.info(
            f"Current violated (%) {np.sum(global_verif_output.status==StatusVerif.VIOLATED)/nb_points*100}"
        )
        logger.info(f"{global_verif_output.runtime_per_block[method.__name__]} sec of computing for block {method.__name__}")
        index_method += 1
    # accuracy
    t_n = time.perf_counter()
    
    return global_verif_output,t_n



def merge_global_verif_outputs(list_global_verif: List[GlobalVerifOutput]) -> GlobalVerifOutput:
    if not list_global_verif or not len(list_global_verif):
        raise ValueError("GlobalVerifOutput object list is empty.")

    # Initialization
    merged_results = []
    merged_status = []
    merged_index_block_that_concluded = []
    merged_inputs = []
    merged_outputs = []
    total_build_time = 0.0
    merged_init_time_per_sample = []
    merged_verif_time_per_sample = []
    merged_runtime_per_block={}

    for global_verif in list_global_verif:
        assert(global_verif.methods==list_global_verif[0].methods)#make sure it's same list for all batches
        merged_results.extend(global_verif.results)
        merged_status.append(global_verif.status)
        
        # Index des blocs qui ont conclu
        merged_index_block_that_concluded.append(global_verif.index_block_that_concluded)
        
        # Inputs et Outputs
        merged_inputs.extend(global_verif.inputs)
        merged_outputs.extend(global_verif.outputs)
        
        # Temps total
        total_build_time += global_verif.build_time
        
        # Temps par sample
        merged_init_time_per_sample.append(global_verif.init_time_per_sample)
        merged_verif_time_per_sample.append(global_verif.verif_time_per_sample)
        for method_name, runtime in global_verif.runtime_per_block.items():
            merged_runtime_per_block[method_name] = (
                merged_runtime_per_block.get(method_name, 0) + runtime
            )
    # Fusionner les tableaux numpy en un seul
    merged_status = np.concatenate(merged_status, axis=0)
    merged_index_block_that_concluded = np.concatenate(merged_index_block_that_concluded, axis=0)
    merged_init_time_per_sample = np.concatenate(merged_init_time_per_sample, axis=0)
    merged_verif_time_per_sample = np.concatenate(merged_verif_time_per_sample, axis=0)
    
    # Retourner un nouvel objet GlobalVerifOutput avec les données fusionnées
    return GlobalVerifOutput(
        methods=list_global_verif[0].methods,#list(set(merged_methods)),
        results=merged_results,
        status=merged_status,
        index_block_that_concluded=merged_index_block_that_concluded,
        inputs=merged_inputs,
        outputs=merged_outputs,
        build_time=total_build_time,
        init_time_per_sample=merged_init_time_per_sample,
        verif_time_per_sample=merged_verif_time_per_sample,
        runtime_per_block=merged_runtime_per_block
    )