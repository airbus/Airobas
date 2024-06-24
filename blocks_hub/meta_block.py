from typing import Dict, List, Tuple, Type

import numpy as np

from verif_pipeline import (
    BlockVerif,
    BlockVerifOutput,
    DataContainer,
    GlobalVerifOutput,
    ProblemContainer,
    full_verification_pipeline,
)


class MetaBlock(BlockVerif):
    def __init__(
        self,
        problem_container: ProblemContainer,
        data_container: DataContainer,
        blocks_verifier: List[Tuple[Type[BlockVerif], Dict]],
    ):
        super().__init__(problem_container, data_container)
        self.block_verifier = blocks_verifier

    def verif(self, indexes: np.ndarray) -> BlockVerifOutput:
        glob: GlobalVerifOutput = full_verification_pipeline(
            problem=self.problem_container,
            input_points=self.data_container.input_points[indexes, :],
            output_points=self.data_container.output_points[indexes, :],
            blocks_verifier=self.block_verifier,
        )
        return BlockVerifOutput(
            status=glob.status,
            inputs=glob.inputs,
            outputs=glob.outputs,
            build_time=glob.build_time,
            init_time_per_sample=glob.init_time_per_sample,
            verif_time_per_sample=glob.verif_time_per_sample,
        )

    @staticmethod
    def get_name() -> str:
        return "meta-block"
