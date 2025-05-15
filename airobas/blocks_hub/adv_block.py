import time
from typing import Dict, List, Optional

import keras.ops as K
import numpy as np
from airobas.blocks_hub.meta_block import MetaBlock
from airobas.verif_pipeline import (
    BlockVerif,
    BlockVerifOutput,
    DataContainer,
    ProblemContainer,
    StatusVerif,
)

# from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
# from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent
from .adversarial.fgsm import fast_gradient_method
from .adversarial.pgd import projected_gradient_descent


def custom_adv_loss(logits, mask_output, mask_target_up):
    """
    Adversarial loss to increase or decrease a specific output prediction.

    This custom loss function computes the adversarial loss that adjusts the model's output
    by increasing or decreasing a specific output prediction, as determined by the provided
    `mask_target_up` and `mask_output`. The goal is to modify the output prediction in a
    targeted manner, either increasing or decreasing it based on the mask.

    Args:
        logits: The model's raw output predictions (logits) before softmax or activation.
        mask_output: A one-hot encoded mask tensor, where the target output's prediction is indicated.
        mask_target_up: A mask that specifies whether to attack by increasing (+1) or decreasing (-1)
                                            the output. Positive values increase the output, while negative values decrease it.

    Returns:
        A Tensor containing the adversarial loss, computed as the sum of the targeted attack on the output predictions.
    """
    # logits = K.convert_to_tensor(logits, dtype='float32')
    mask_target_up = K.convert_to_tensor(mask_target_up, dtype="float32")
    mask_output = K.convert_to_tensor(mask_output, dtype="float32")
    return K.sum(mask_output * (logits * K.expand_dims(mask_target_up, -1)))


def check_SB_sat(Y_pred, Y_min, Y_max):
    """
    Checks whether predicted outputs violate the specified output stability bounds.

    This function evaluates if each predicted output vector (`Y_pred`) lies entirely
    within the specified bounds given by `Y_min` and `Y_max`. If a prediction violates
    the bounds (either exceeding `Y_max` or falling below `Y_min`), it is marked as unstable.

    Args:
        Y_pred: Predicted output from the model. Shape: (n_samples, n_outputs).
        Y_min: Lower bounds for each output dimension. Same shape as `Y_pred`.
        Y_max: Upper bounds for each output dimension. Same shape as `Y_pred`.

    Returns:
        np.ndarray: A one-hot encoded array of shape (n_samples, 2), where:
            - `[1, 0]` indicates a **stable** prediction (within bounds),
            - `[0, 1]` indicates an **unstable** prediction (satisfies SAT condition, i.e., violates bounds).
    """

    labels = np.zeros((len(Y_pred), 2))

    dist_up = np.max(Y_pred - Y_max, -1)  # if positive, sat Y_max<= Y_pred
    dist_low = np.min(Y_pred - Y_min, -1)  # if negative, sat Y_pred<=Y_min
    labels[np.where(dist_up >= 0)[0], 0] = 1
    labels[np.where(dist_low <= 0)[0], 0] = 1

    return labels


def adv_func_priv(model, X_min, X_max, Y_min, Y_max, loss_fn, fgs=True, target_index=None, preds=False, **kwargs):
    """
    Generate adversarial samples constrained within input stability bounds using
    either Fast Gradient Sign Method (FGSM) or Projected Gradient Descent (PGD).

    Args:
        model: Trained Keras model used for generating adversarial examples.
        X_min: Lower bounds of the input domain.
        X_max: Upper bounds of the input domain.
        Y_min: Lower bounds for the expected output stability range.
        Y_max: Upper bounds for the expected output stability range.
        loss_fn: Loss function to use for crafting the adversarial examples.
        fgs: If True, uses FGSM; if False, uses PGD. Default is True.
        target_index: Not currently used in this function, but could be used for targeted attacks.
        preds: If True, also return adversarial inputs and model predictions. Default is False.
        **kwargs: Additional arguments:
            - nb_iter (int): Number of iterations for PGD attack (only used if fgs is False).

    Returns:
        If preds == False:
            One-hot encoded labels of shape (n_samples, 2)
                        where [0, 1] means unstable (violation) and [1, 0] means stable.
        If preds == True:
                - One-hot encoded labels as above.
                - List of tuples: (adversarial input, predicted output)
    """
    X = (X_min + X_max) / 2.0
    ts = time.perf_counter()
    # update_summary(kwargs.get("summary_writer", None), timestamp=ts, value=0, key="runtime")
    # one hot encoding for masking the output that will not be targeted

    eps = max(np.abs(X_min).max(), np.abs(X_max).max())
    if fgs:
        # run fast gradient sign
        X_adv = fast_gradient_method(
            model,
            K.convert_to_tensor(X, dtype="float32"),
            eps=eps,
            norm=np.inf,
            loss_fn=loss_fn,
            clip_min=K.convert_to_tensor(X_min, dtype="float32"),
            clip_max=K.convert_to_tensor(X_max, dtype="float32"),
            y=Y_min,
        )

    else:
        eps_iter = 0.001
        nb_iter = kwargs.get("nb_iter", 400)
        X_adv = projected_gradient_descent(
            model,
            K.convert_to_tensor(X, dtype="float32"),
            eps=eps,
            eps_iter=eps_iter,
            nb_iter=nb_iter,
            norm=np.inf,
            loss_fn=loss_fn,
            clip_min=K.convert_to_tensor(X_min, dtype="float32"),
            clip_max=K.convert_to_tensor(X_max, dtype="float32"),
            y=Y_min,
        )

    # determine the bounds
    Y_pred = model.predict(X_adv, verbose=0)
    X_adv = X_adv.cpu().detach()
    # Y_min, Y_max = discard_untarget_outputs_verif(Y_min, Y_max, target_index, model.output_shape[-1])
    labels = check_SB_sat(Y_pred, Y_min, Y_max)
    if preds:
        return labels, [(X_adv[i, :].numpy(), Y_pred[i, :]) for i in range(X_adv.shape[0])]  # Y_pred
    else:
        return labels


def get_adv_func(model, index_target, up, fgs=True, preds=False, **kwargs):
    """Tuning the generation of adversarial attacks

    Args:
        index_target :int for which output is targeter
        up : boolean set to True for attacks by increasing the output values
        fgs : boolean set to True for FGS and False for PGD attacks

    Returns:
        the adversarial attack function
    """
    mask_target_up = np.ones((1, 1), dtype="float32")
    if not up:
        mask_target_up *= -1

    # generate random target
    output_dim = model.output_shape[-1]
    mask_adv = np.zeros((1, output_dim), dtype="float32")
    mask_adv[:, index_target] = 1

    def loss_fn(logits, labels):
        return custom_adv_loss(logits, mask_adv[None], mask_target_up[None])

    def adv_(model, X_min, X_max, Y_min, Y_max, target_index=None):
        """adversarial attacks for validation (can only found unstable samples)

        Args:
            model: Keras model
            X_min : lower bound of the input domain to ensure stability
            X_max : upper bound of the input domain to ensure stability
            Y_min : lower bound of the output domain to ensure stability
            Y_max : upper bound of the output domain to ensure stability

        Returns:
            one hot encoding with class 0 for unstable samples (sat)
        """
        return adv_func_priv(
            model, X_min, X_max, Y_min, Y_max, loss_fn, fgs=fgs, target_index=target_index, preds=preds, **kwargs
        )

    return adv_


class CleverhansAdvBlock(BlockVerif):
    """
    Verification block based on adversarial example generation using CleverHans attacks.

    This block attempts to find violations of a model's output stability property by generating
    adversarial examples within a specified input domain. It uses Fast Gradient Sign Method (FGS)
    or Projected Gradient Descent (PGD) to perturb inputs and evaluate whether the model's outputs
    exceed the allowed output bounds.

    Parameters:
        problem_container:
            The container holding the model and property to verify.

        data_container:
            The container holding input bounds (X_min, X_max) and corresponding output bounds (Y_min, Y_max)
            for the current batch.

        index_target:
            Index of the output dimension to target for the attack. This allows focused attacks on a specific output.

        attack_up:
            Whether to perform the attack by increasing (`True`) or decreasing (`False`) the targeted output value.

        fgs:
            If `True`, use Fast Gradient Sign (FGS) method. If `False`, use Projected Gradient Descent (PGD).


    Notes:
        - Violations are determined using the `check_SB_sat` function, which checks whether the adversarial
          prediction falls outside of the given output stability bounds (Y_min, Y_max).
        - Timing is reported per-sample uniformly for the violating samples.
        - Only violated samples are recorded as counterexamples in the output.
        - This block does not have a separate model building phase, so `build_time` is always zero.

    Example:
        block = CleverhansAdvBlock(problem, data, index_target=0, attack_up=True, fgs=True)
        result = block.verif(np.arange(len(data.input_points)))
    """

    def __init__(
        self,
        problem_container: ProblemContainer,
        data_container: DataContainer,
        index_target: int,
        attack_up: bool,
        fgs: bool,
    ):
        super().__init__(problem_container=problem_container, data_container=data_container)
        self.method = get_adv_func(
            self.problem_container.model,
            index_target=index_target,
            up=attack_up,
            fgs=fgs,
            preds=True,
        )
        self.params = {"index_target": index_target, "attack_up": attack_up, "fgs": fgs}

    def verif(self, indexes: np.ndarray) -> BlockVerifOutput:
        nb_points = len(indexes)
        output = BlockVerifOutput(
            status=np.array([StatusVerif.UNKNOWN for _ in range(nb_points)], dtype=StatusVerif),
            inputs=[None for _ in range(nb_points)],
            outputs=[None for _ in range(nb_points)],
            build_time=0,
            init_time_per_sample=np.empty(nb_points, dtype=float),
            verif_time_per_sample=np.empty(nb_points, dtype=float),
        )
        t1 = time.perf_counter()
        labels, predictions = self.method(
            model=self.problem_container.model,
            X_min=self.data_container.lbound_input_points[indexes, :],
            X_max=self.data_container.ubound_input_points[indexes, :],
            Y_min=self.data_container.lbound_output_points[indexes, :],
            Y_max=self.data_container.ubound_output_points[indexes, :],
            target_index=None,
        )
        t2 = time.perf_counter()
        unsat = np.nonzero(labels[:, 0])
        output.build_time = 0  # No special build time for cleverhans
        output.verif_time_per_sample[unsat] = t2 - t1
        output.init_time_per_sample[unsat] = 0
        output.status[unsat] = StatusVerif.VIOLATED
        for i in unsat[0]:
            output.inputs[i] = predictions[i][0]
            output.outputs[i] = predictions[i][1]
        return output

    @staticmethod
    def get_name() -> str:
        return "cleverhans-adv-block"


class CleverHansMultiIndexAdvBlock(MetaBlock):
    """
    A meta verification block that applies multiple `CleverhansAdvBlock` instances with different configurations.

    This block aggregates multiple adversarial verification strategies to comprehensively evaluate the
    robustness of a model across all output indices and attack directions (increase or decrease), using
    adversarial attacks from the CleverHans library (FGS or PGD).

    Parameters:
        problem_container:
            Container that holds the model and the property to be verified.

        data_container:
            Container with the input and output bounds used during verification.

        list_params_adv_block:
            Optional list of dictionaries, each specifying parameters for a `CleverhansAdvBlock`:
            - "index_target" (int): which output index to target
            - "attack_up" (bool): whether to try increasing the target output
            - "fgs" (bool): whether to use Fast Gradient Sign (True) or PGD (False)

            If not provided, a default configuration is generated which includes:
                - All output indices (`range(output_dim)`)
                - Both attack directions (`True` and `False`)
                - Only FGS (`fgs=True`)


    Example:
        >>> meta_block = CleverHansMultiIndexAdvBlock(problem, data)
        >>> result = meta_block.verif(np.arange(len(data.input_points)))

    Notes:
        - This block creates a full coverage adversarial strategy over all output components.
        - It is particularly useful for evaluating multi-output models (e.g., regression or control systems).
        - Runtime can be high due to the large number of sub-blocks if output dimensionality is large.
        - Can be used with batching by `full_verification_pipeline` for scalability.
    """

    def __init__(
        self,
        problem_container: ProblemContainer,
        data_container: DataContainer,
        list_params_adv_block: Optional[List[Dict]] = None,
    ):
        if list_params_adv_block is None:
            list_params_adv_block = [
                {"index_target": i, "attack_up": b, "fgs": c}
                for c in [True]  # ,False]
                for i in range(data_container.output_points.shape[1])
                for b in [True, False]
            ]
        super().__init__(
            problem_container,
            data_container,
            blocks_verifier=[(CleverhansAdvBlock, d) for d in list_params_adv_block],
        )

    @staticmethod
    def get_name() -> str:
        return "meta-cleverhans-block"
