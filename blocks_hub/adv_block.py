import time
from typing import Dict, List, Optional

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent

from blocks_hub.meta_block import MetaBlock
from verif_pipeline import (
    BlockVerif,
    BlockVerifOutput,
    DataContainer,
    ProblemContainer,
    StatusVerif,
)


def custom_adv_loss(logits, mask_output, mask_target_up):
    """adversarial loss to increase/decrease a specific output prediction

    Args:
        logits: output prediction
        mask_output: one hot encoding to attack a specific output
        mask_target_up: mask to target whether we attack by increasing (+1) or decreasing (-1) the output

    Returns:
        adversarial loss
    """
    return K.sum(mask_output * (logits * K.expand_dims(mask_target_up, -1)), -1)


def check_SB_sat(Y_pred, Y_min, Y_max):
    """

    Args:
        Y_pred : model prediction
        Y_min: lower bound of the output domain to ensure stability
        Y_max: upper bound of the output domain to ensure stability

    Returns:
        one hot encoding with class 0 for unstable samples (sat)
    """

    labels = np.zeros((len(Y_pred), 2))

    dist_up = np.max(Y_pred - Y_max, -1)  # if positive, sat Y_max<= Y_pred
    dist_low = np.min(Y_pred - Y_min, -1)  # if negative, sat Y_pred<=Y_min
    labels[np.where(dist_up >= 0)[0], 0] = 1
    labels[np.where(dist_low <= 0)[0], 0] = 1

    return labels


def adv_func_priv(
    model,
    X_min,
    X_max,
    Y_min,
    Y_max,
    loss_fn,
    fgs=True,
    target_index=None,
    preds=False,
    **kwargs
):
    """Generate adversarial samples within the stable input bounds (with FGS or PGD)

    Args:
        model: Keras model
        X : inputs
        y : groundtruth labels
        attack_up : Proportion of attacks by increasing the output values
        target : 'GT' for using the GroundTruth labels else clip the model prediction to the sability bounds
        fgs : boolean set to True for FGS and False for PGD attacks

    Returns:
        augmented inputs and outputs
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
            tf.constant(X),
            eps=eps,
            norm=np.inf,
            loss_fn=loss_fn,
            clip_min=X_min,
            clip_max=X_max,
            y=Y_min,
        )

    else:
        eps_iter = 0.001
        nb_iter = kwargs.get("nb_iter", 400)
        X_adv = projected_gradient_descent(
            model,
            tf.constant(X),
            eps=eps,
            eps_iter=eps_iter,
            nb_iter=nb_iter,
            norm=np.inf,
            loss_fn=loss_fn,
            clip_min=X_min,
            clip_max=X_max,
            y=Y_min,
        )

    # determine the bounds
    Y_pred = model.predict(X_adv, verbose=0)
    # Y_min, Y_max = discard_untarget_outputs_verif(Y_min, Y_max, target_index, model.output_shape[-1])
    labels = check_SB_sat(Y_pred, Y_min, Y_max)
    if preds:
        return labels, [
            (X_adv[i, :].numpy(), Y_pred[i, :]) for i in range(X_adv.shape[0])
        ]  # Y_pred
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

    def loss_fn(labels, logits):
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
            model,
            X_min,
            X_max,
            Y_min,
            Y_max,
            loss_fn,
            fgs=fgs,
            target_index=target_index,
            preds=preds,
            **kwargs
        )

    return adv_


class CleverhansAdvBlock(BlockVerif):
    def __init__(
        self,
        problem_container: ProblemContainer,
        data_container: DataContainer,
        index_target: int,
        attack_up: bool,
        fgs: bool,
    ):
        super().__init__(
            problem_container=problem_container, data_container=data_container
        )
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
            status=np.array(
                [StatusVerif.UNKNOWN for _ in range(nb_points)], dtype=StatusVerif
            ),
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
    def __init__(
        self,
        problem_container: ProblemContainer,
        data_container: DataContainer,
        list_params_adv_block: Optional[List[Dict]] = None,
    ):
        if list_params_adv_block is None:
            list_params_adv_block = [
                {"index_target": i, "attack_up": b, "fgs": True}
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
