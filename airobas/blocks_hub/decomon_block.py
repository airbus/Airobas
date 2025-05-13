import time

import numpy as np
from airobas.verif_pipeline import BlockVerif, BlockVerifOutput, StatusVerif
from decomon.models import clone


def check_SB_unsat(y_pred_min, y_pred_max, y_min, y_max):
    """

    Args:
        y_pred_min : lower bound for incomplete certificate (decomon, lirpa)
        y_pred_max : upper bound for incomplete certificate (decomon, lirpa)
        y_min: lower bound of the output domain to ensure stability
        y_max: upper bound of the output domain to ensure stability

    Returns:
        one hot encoding with class 1 for stable samples (unsat)
    """

    labels_max = np.zeros((len(y_min),))
    labels_min = np.zeros((len(y_min),))
    labels = np.zeros((len(y_min), 2))

    dist_up = np.max(y_pred_max - y_max, -1)  # should be negative
    dist_low = np.min(y_pred_min - y_min, -1)  # should be positive

    labels_max[np.where(dist_up <= 0)[0]] = 1
    labels_min[np.where(dist_low >= 0)[0]] = 1

    labels[:, 1] = labels_max * labels_min

    return labels


class DecomonBlock(BlockVerif):
    @staticmethod
    def get_name() -> str:
        return "decomon-block"

    def verif(self, indexes: np.ndarray) -> BlockVerifOutput:
        nb_points = len(indexes)
        output = BlockVerifOutput(
            status=np.array([StatusVerif.UNKNOWN for i in range(nb_points)], dtype=StatusVerif),
            inputs=[None for i in range(nb_points)],
            outputs=[None for i in range(nb_points)],
            build_time=0,
            init_time_per_sample=np.empty(nb_points, dtype=float),
            verif_time_per_sample=np.empty(nb_points, dtype=float),
        )
        x_min = self.data_container.lbound_input_points[indexes, :]
        x_max = self.data_container.ubound_input_points[indexes, :]
        t1 = time.perf_counter()
        decomon_model = clone(self.problem_container.model)
        output.build_time = time.perf_counter() - t1
        box = np.concatenate([x_min[:, None], x_max[:, None]], 1)
        t2 = time.perf_counter()
        y_up, y_low = decomon_model.predict(box)
        labels = check_SB_unsat(
            y_pred_min=y_low,
            y_pred_max=y_up,
            y_min=self.data_container.lbound_output_points[indexes, :],
            y_max=self.data_container.ubound_output_points[indexes, :],
        )
        t3 = time.perf_counter()
        indexes = np.nonzero(labels[:, 1])
        output.status[indexes] = StatusVerif.VERIFIED  # this method only conclude on "robust" points
        output.init_time_per_sample[indexes] = t2 - t1
        output.verif_time_per_sample[indexes] = t3 - t2
        return output
