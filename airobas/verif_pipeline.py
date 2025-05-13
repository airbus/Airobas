import logging
import time
from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Type, Union

# from .utils import merge_global_verif_outputs
import keras
import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


class StatusVerif(Enum):
    """
    Enum representing the verification status of a property during analysis.

    Attributes:
        VERIFIED (int): The property has been successfully verified.
        VIOLATED (int): The property was violated during analysis.
        UNKNOWN (int): The result is inconclusive, typically due to an incomplete method.
        TIMEOUT (int): The analysis timed out, often due to resource or time limitations in complete methods.
    """

    VERIFIED = 0
    """The property has been successfully verified to hold in all cases."""
    VIOLATED = 1
    """A counterexample was found; the property does not always hold."""
    UNKNOWN = 2  # incomplete methods cannot conclude
    """Analysis was inconclusive, usually due to incomplete methods."""
    TIMEOUT = 3  # this is more for complete methods that runs out of time.
    """Analysis did not complete in time; applicable to complete methods."""


class BoundsDomainParameter:
    """
    Abstract base class for domain parameters that define methods
    to compute lower and upper bounds for given inputs.

    This class is intended to be subclassed with concrete implementations
    of the `compute_lb_ub_bounds` method.
    """

    @abstractmethod
    def compute_lb_ub_bounds(self, x: npt.NDArray) -> npt.NDArray:
        """
        Compute the lower and upper bounds for a given input array.

        Args:
            x: Input array for which bounds are to be computed.

        Returns:
            An array containing the computed lower and upper bounds.
                         The shape and semantics of the output depend on the subclass implementation.
        """
        ...


@dataclass
class StabilityProperty:
    """
    Represents a stability property defined by input and output bound domains.

    This class encapsulates the parameters required to describe a stability
    condition, where both input and output bounds are determined using
    instances of `BoundsDomainParameter`.

    Attributes:
        input_bound_domain_param : The domain parameter used to compute bounds on the input space.

        output_bound_domain_param: The domain parameter used to compute bounds on the output space.
    """

    input_bound_domain_param: BoundsDomainParameter
    output_bound_domain_param: BoundsDomainParameter


@dataclass
class BoundsDomainBoxParameter(BoundsDomainParameter):
    """
    A concrete implementation of BoundsDomainParameter that defines
    bounds using either relative or absolute noise around input values.

    Attributes:
        rel_noise: Relative noise factor applied to each input element (e.g., 0.1 for ±10%). Used only if `use_relative` is True.

        abs_noise: Absolute noise value added/subtracted from each input element.
            Used only if `use_relative` is False.

        use_relative : Flag indicating whether to use relative (`True`) or absolute (`False`) noise.
    """

    rel_noise: Optional[float] = None
    abs_noise: Optional[float] = None
    use_relative: bool = True

    def compute_lb_ub_bounds(self, x: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
        """
        Compute the lower and upper bounds for the input array `x`.

        If `use_relative` is True, the bounds are computed as:
            lower = x - rel_noise * |x|
            upper = x + rel_noise * |x|

        Otherwise, the bounds are computed as:
            lower = x - abs_noise
            upper = x + abs_noise

        Args:
            x : The input array for which to compute bounds.

        Returns:
            A tuple containing the lower and upper bound arrays.

        Raises:
            If the appropriate noise parameter (`rel_noise` or `abs_noise`) is not set depending on the mode.
        """
        if self.use_relative:
            return x - self.rel_noise * np.abs(x), x + self.rel_noise * np.abs(x)
        return x - self.abs_noise, x + self.abs_noise


@dataclass
class BoundsDomainBoxParameterPerValueInterval(BoundsDomainParameter):
    """
    Applies different bounding strategies depending on the value intervals
    of the input elements.

    Each tuple in `bounds_domain_parameters` defines a value interval (val1, val2]
    and an associated `BoundsDomainParameter` that will be used for computing
    the bounds of input values falling within that interval.

    Attributes:
        bounds_domain_parameters:
            A list of tuples where each tuple contains:
              - lower bound (exclusive),
              - upper bound (inclusive),
              - a `BoundsDomainParameter` used to compute bounds within that interval.
    """

    bounds_domain_parameters: List[Tuple[float, float, BoundsDomainParameter]]

    def compute_lb_ub_bounds(self, x: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
        """
        Computes element-wise lower and upper bounds for the input array `x`,
        using the bound strategy assigned to each input interval.

        For each value in `x`, determines the corresponding interval in
        `bounds_domain_parameters` and applies the appropriate bounding method.

        Args:
            x: Input array of values to compute bounds for.

        Returns:
            Two arrays (lower bounds, upper bounds) with the same shape as `x`, containing the computed bounds.

        Notes:
            - Values not falling into any of the defined intervals will retain
              uninitialized values in the output (e.g., `np.empty` default behavior).
            - It is the caller's responsibility to ensure the intervals cover all
              possible values in `x` if full coverage is desired.
        """
        merge_lb, merge_ub = np.empty(x.shape), np.empty(x.shape)
        for val1, val2, bparameters in self.bounds_domain_parameters:
            indexes = np.nonzero(np.logical_and(val1 < x, x <= val2))
            lb, ub = bparameters.compute_lb_ub_bounds(x[indexes])
            merge_lb[indexes] = lb
            merge_ub[indexes] = ub
        return merge_lb, merge_ub


@dataclass
class ProblemContainer:
    """
    Container for encapsulating a verification problem involving a model
    and its associated stability property.

    Attributes:
        tag_id : An identifier for the problem instance, useful for tracking or logging.

        model: The machine learning model under analysis.

        stability_property: The stability property specifying how to bound the model's input and output.
    """

    def __init__(
        self,
        tag_id: Union[str, int],
        model: keras.Model,
        stability_property: StabilityProperty,
    ):
        """
        Initialize a ProblemContainer.

        Args:
            tag_id:
                A unique identifier for the problem (e.g., an index or descriptive label).

            model:
                The neural network model to be analyzed.

            stability_property:
                Describes how input/output perturbations should be handled in the verification task.
        """
        self.tag_id = tag_id
        self.model = model
        self.stability_property = stability_property


@dataclass
class DataContainer:
    """
    Utility class to store data used in experiments, including input/output points
    and their corresponding lower and upper bounds.

    Attributes:
        input_points:
            The original input samples used in the experiment.

        lbound_input_points:
            The lower bounds for each input sample.

        ubound_input_points:
            The upper bounds for each input sample.

        output_points:
            The model's output values for each input sample.

        lbound_output_points:
            The lower bounds for each output prediction.

        ubound_output_points:
            The upper bounds for each output prediction.
    """

    def __init__(
        self,
        input_points: np.ndarray,
        lbound_input_points: np.ndarray,
        ubound_input_points: np.ndarray,
        output_points: np.ndarray,
        lbound_output_points: np.ndarray,
        ubound_output_points: np.ndarray,
    ):
        """
        Initialize a DataContainer instance with input/output data and bounds.

        Args:
            input_points:
                Input samples used in the experiment.

            lbound_input_points:
                Lower bounds for the input samples.

            ubound_input_points:
                Upper bounds for the input samples.

            output_points:
                Output values (e.g., predictions) corresponding to input points.

            lbound_output_points:
                Lower bounds for output predictions.

            ubound_output_points:
                Upper bounds for output predictions.
        """
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
    """
    Compute lower and upper bounds for the given array `x` based on a stability property.

    Depending on the `is_input` flag, this function uses either the input or output
    bound domain parameter from the provided `StabilityProperty`.

    Args:
        stability_property (StabilityProperty):
            The stability property defining how bounds are computed.

        x (np.ndarray):
            The array of input or output values for which to compute bounds.

        is_input (bool, optional):
            If True, compute bounds using the input bound domain parameter;
            otherwise, use the output bound domain parameter. Defaults to True.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            A tuple containing the lower and upper bounds as NumPy arrays, matching the shape of `x`.
    """
    if is_input:
        return stability_property.input_bound_domain_param.compute_lb_ub_bounds(x)
    else:
        return stability_property.output_bound_domain_param.compute_lb_ub_bounds(x)


@dataclass
class BlockVerifOutput:
    """
    Container for the output of a block-level verification process.

    This class stores the verification status, counterexamples (if any),
    and timing information related to the verification workflow.

    Attributes:
        status:
            An array indicating the verification status for each sample,
            typically using values from the `StatusVerif` enum.

        inputs:
            A list containing counterexample input arrays for each sample.
            If no counterexample is found, the corresponding entry is `None`.

        outputs:
            A list containing counterexample output arrays for each sample.
            If no counterexample is found, the corresponding entry is `None`.

        build_time:
            Total time taken for initial setup, such as loading models or allocating resources.

        init_time_per_sample:
            Array of times representing model compilation or setup duration for each sample,
            prior to executing verification queries.

        verif_time_per_sample:
            Array of times representing the actual verification duration for each sample,
            after model initialization.
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
    """
    Aggregated output of a full verification pipeline across multiple blocks/methods.

    This class captures detailed verification results from multiple verification blocks,
    their statuses, timings, and counterexamples—providing a global summary of the process.

    Attributes:
        methods: List of verification method names, one per block.

        results:
            Detailed results for each verification block.
            Each tuple contains:
              - A BlockVerifOutput instance.
              - An optional list of original indices of the samples handled by that block.

        status:
            Final verification status for the full dataset. Shape matches the dataset size.

        index_block_that_concluded:
            Array indicating, for each sample, the index of the block/method
            that provided the final verification result.

        inputs:
            Counterexample input points for each sample, if any; `None` if none was found.

        outputs:
            Counterexample output predictions for each sample, if any; `None` if none was found.

        build_time:
            Total build time across all verification blocks (e.g., model loading/setup).

        init_time_per_sample:
            Per-sample model initialization times aggregated from all blocks.

        verif_time_per_sample:
            Per-sample verification times aggregated from all blocks.

        runtime_per_block:
            Total runtime per verification method name, aggregated from all blocks.
    """

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
    runtime_per_block: Dict[str, float]  # sum of runtime of verification per method/block


class BlockVerif:
    """
    Abstract base class representing a block-level verification process.

    This class is responsible for performing the verification task for a given
    problem and data container, where each verification block handles a specific
    task or subset of verification. Concrete subclasses should implement the `verif`
    method to define the block-specific verification logic.

    Attributes:
        problem_container:
            The container holding the problem information, including the model and stability property.

        data_container:
            The container holding the input/output data and their bounds for verification.
    """

    def __init__(self, problem_container: ProblemContainer, data_container: DataContainer):
        """
        Initialize a BlockVerif instance with a given problem and data container.

        Args:
            problem_container:
                Contains the model and its associated stability properties.

            data_container:
                Contains the input/output points and their bounds used for verification.
        """
        self.problem_container = problem_container
        self.data_container = data_container

    @abstractmethod
    def verif(self, indexes: np.ndarray) -> BlockVerifOutput:
        """
        Perform the verification for the given set of indexes.

        This method should be implemented in subclasses to execute the block-level
        verification process, using the `problem_container` and `data_container`
        information. The `indexes` array specifies which data points to verify.

        Args:
            indexes:
                Array of indices representing the samples to be verified within the block.

        Returns:
            BlockVerifOutput:
                The output of the block-level verification, containing status, counterexamples,
                and timing information for the verification process.
        """
        ...

    @staticmethod
    def get_name() -> str:
        """
        Get the name of the verification block.

        This method should be implemented in subclasses to return a string representing
        the name or identifier of the block verification method.

        Returns:
            str:
                The name of the verification block.
        """
        ...


def full_verification_pipeline(
    problem: ProblemContainer,
    input_points: np.ndarray,
    # data points to test property on
    output_points: Optional[np.ndarray],
    # could be model inference of input_points, or ground truth etc.
    blocks_verifier: List[Union[Type[BlockVerif], Dict]],
    verbose: bool = True,
    batch_split: int = 1,
) -> GlobalVerifOutput:
    """
    Executes the full verification pipeline for a given model and stability property.

    This function runs verification in batches, utilizing specified verification blocks,
    and computes lower and upper bounds for both inputs and outputs. If no `output_points`
    are provided, the model will be used to generate them. The function returns a summary
    of the verification results.

    Args:
        problem:
            The container that holds the problem configuration, including the model and its stability property.

        input_points:
            The data points for which the verification property is to be tested.

        output_points:
            The expected output values for each input point (e.g., model predictions or ground truth).
            If not provided, the model's predictions are computed.

        blocks_verifier:
            A list of block verification methods to run, each either being a class or a dictionary.

        verbose:
            If True, prints detailed progress and debugging information during the process. Defaults to True.

        batch_split:
            The number of batches to split the dataset into for parallel processing. Defaults to 1 (no batching).

    Returns:
        An object containing the aggregated results of the verification process across all blocks and batches,
        including statuses, counterexamples, timings, and runtime summaries.
    """

    if output_points is None:
        output_points = problem.model.predict(input_points, verbose=0)
    t_0 = time.perf_counter()

    x_min, x_max = compute_bounds(problem.stability_property, input_points, is_input=True)
    # Bounds for input ..
    y_min, y_max = compute_bounds(problem.stability_property, output_points, is_input=False)
    # loop for batches
    nb_samples = x_min.shape[0]  # here we get nb_samples, with one sample it's always one?
    batch_size = nb_samples // batch_split + 1

    if nb_samples % batch_split != 0:  # Handle uneven division
        batch_size += 1
    list_global_verif = []
    t_n = 0
    runtime_summary = {}  # key: method.__name__, value: total runtime in sec
    for i in range(batch_split):
        if verbose:
            print(f"Batch number {i}")
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, nb_samples)  # Ensure end_idx does not exceed array length

        input_sample_i = input_points[start_idx:end_idx]
        output_sample_i = output_points[start_idx:end_idx]

        x_min_i = x_min[start_idx:end_idx]
        x_max_i = x_max[start_idx:end_idx]

        y_min_i = y_min[start_idx:end_idx]
        y_max_i = y_max[start_idx:end_idx]

        assert len(input_sample_i) == len(
            output_sample_i
        ), f"Mismatch between input and output batch sizes: {len(input_sample_i)} vs {len(output_sample_i)}"

        if len(input_sample_i) == 0:
            print(f"Batch {i + 1} is empty. Breaking.")
            break
        if verbose:
            print(f"Processing batch {i + 1}/{batch_split}:")
            print(f"Start index: {start_idx}, End index: {end_idx}")
            print(f"Input batch shape: {input_sample_i.shape}")
            print(f"Output batch shape: {output_sample_i.shape}")

        global_verif_output_i, t_n_i = full_verification_pipeline_batch(
            0, input_sample_i, x_min_i, x_max_i, output_sample_i, y_min_i, y_max_i, blocks_verifier, problem
        )
        t_n += t_n_i
        list_global_verif.append(global_verif_output_i)
    global_verif_output: GlobalVerifOutput = merge_global_verif_outputs(list_global_verif)
    logger.info(f"Total time of verif : {t_n-t_0} seconds")
    return global_verif_output


def full_verification_pipeline_batch(
    index_batch, input_points, x_min, x_max, output_points, y_min, y_max, blocks_verifier, problem
):
    """
    Performs verification for a batch of input points using a list of block verifiers.

    This function processes a batch of input points and runs verification using multiple
    blocks defined in `blocks_verifier`. Each block is responsible for verifying a certain
    property, and the results are accumulated in the `GlobalVerifOutput` object. The function
    also computes the runtime for each block and updates the overall results for the batch.

    Args:
        index_batch:
            The index number of the current batch being processed. This is used for logging purposes.

        input_points:
            The input points to be verified.

        x_min:
            The lower bounds for each input point, used to compute the verification results.

        x_max:
            The upper bounds for each input point, used to compute the verification results.

        output_points:
            The expected output points corresponding to the input points (e.g., model predictions or ground truth).

        y_min:
            The lower bounds for the output points.

        y_max:
            The upper bounds for the output points.

        blocks_verifier:
            A list of tuples containing the verification block classes and their respective keyword arguments.
            Each block will be executed sequentially to verify the input data.

        problem:
            The container holding the problem's configuration, including the model and stability property.

    Returns:
        The first element is a `GlobalVerifOutput` containing the aggregated results of the verification process,
        and the second element is the total time spent processing the batch.

    """
    # Bounds for desired output
    data = DataContainer(input_points, x_min, x_max, output_points, y_min, y_max)
    nb_points = x_min.shape[0]
    global_verif_output = GlobalVerifOutput(
        methods=[b[0].__name__ for b in blocks_verifier],
        results=[],
        status=np.array([StatusVerif.UNKNOWN for _ in range(nb_points)], dtype=StatusVerif),
        index_block_that_concluded=np.empty(nb_points, dtype=int),
        inputs=[None for _ in range(nb_points)],
        outputs=[None for _ in range(nb_points)],
        build_time=0,
        init_time_per_sample=np.empty(nb_points, dtype=float),
        verif_time_per_sample=np.empty(nb_points, dtype=float),
        runtime_per_block={},
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
        global_verif_output.runtime_per_block[method.__name__] = (
            t_end - t_start
        )  # key: method.__name__, value: runtime in sec
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
        logger.info(f"Current verified (%) {np.sum(global_verif_output.status==StatusVerif.VERIFIED)/nb_points*100}")
        logger.info(f"Current violated (%) {np.sum(global_verif_output.status==StatusVerif.VIOLATED)/nb_points*100}")
        logger.info(
            f"{global_verif_output.runtime_per_block[method.__name__]} sec of computing for block {method.__name__}"
        )
        index_method += 1
    # accuracy
    t_n = time.perf_counter()

    return global_verif_output, t_n


def merge_global_verif_outputs(list_global_verif: List[GlobalVerifOutput]) -> GlobalVerifOutput:
    """
    Merges the outputs from multiple GlobalVerifOutput objects into a single GlobalVerifOutput.

    This function takes a list of `GlobalVerifOutput` objects and merges their individual results
    (e.g., status, inputs, outputs, runtime) into a single, consolidated `GlobalVerifOutput`.
    The merged data includes results from all verification batches, aggregated appropriately.
    It also ensures that all batches have consistent methods and merges the associated times (build, initialization, verification).

    Args:
        list_global_verif:
            A list of `GlobalVerifOutput` objects, each representing the results of a verification batch.
            All objects in the list should have the same verification methods.

    Returns:
        A single `GlobalVerifOutput` object containing the merged results from all batches. This includes:
            - `methods`: The list of verification methods used (assumed to be the same across all batches).
            - `results`: The merged verification results, including inputs, outputs, and statuses for each data point.
            - `status`: A concatenated numpy array containing the verification status for all input points.
            - `index_block_that_concluded`: A concatenated array of indices indicating which verification block concluded for each point.
            - `inputs`: A merged list of input points for each verification batch.
            - `outputs`: A merged list of outputs for each verification batch.
            - `build_time`: The total build time across all batches.
            - `init_time_per_sample`: The initialization time per sample for all batches.
            - `verif_time_per_sample`: The verification time per sample for all batches.
            - `runtime_per_block`: A dictionary mapping verification method names to the accumulated runtime across all batches.

    Raises:
        ValueError: If the input list of `GlobalVerifOutput` objects is empty or contains invalid objects.
        AssertionError: If the methods of the `GlobalVerifOutput` objects in the list do not match.

    """
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
    merged_runtime_per_block = {}

    for global_verif in list_global_verif:
        assert global_verif.methods == list_global_verif[0].methods  # make sure it's same list for all batches
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
            merged_runtime_per_block[method_name] = merged_runtime_per_block.get(method_name, 0) + runtime
    # Fusionner les tableaux numpy en un seul
    merged_status = np.concatenate(merged_status, axis=0)
    merged_index_block_that_concluded = np.concatenate(merged_index_block_that_concluded, axis=0)
    merged_init_time_per_sample = np.concatenate(merged_init_time_per_sample, axis=0)
    merged_verif_time_per_sample = np.concatenate(merged_verif_time_per_sample, axis=0)

    # Retourner un nouvel objet GlobalVerifOutput avec les données fusionnées
    return GlobalVerifOutput(
        methods=list_global_verif[0].methods,  # list(set(merged_methods)),
        results=merged_results,
        status=merged_status,
        index_block_that_concluded=merged_index_block_that_concluded,
        inputs=merged_inputs,
        outputs=merged_outputs,
        build_time=total_build_time,
        init_time_per_sample=merged_init_time_per_sample,
        verif_time_per_sample=merged_verif_time_per_sample,
        runtime_per_block=merged_runtime_per_block,
    )
