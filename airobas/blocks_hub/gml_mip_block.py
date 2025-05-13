import logging
import time
from typing import Optional

import gurobipy as gp
import numpy as np
from airobas.blocks_hub.mip_blocks_lib.commons.bounds_propagation.bounds_computation_utils import (
    create_object_bounds,
)
from airobas.blocks_hub.mip_blocks_lib.commons.formula import (
    GT,
    LT,
    NAryConjFormula,
    StateCoordinate,
    VarConstConstraint,
)
from airobas.blocks_hub.mip_blocks_lib.commons.neural_network import (
    InputBoundsNeuralNetwork,
    NeuralNetwork,
)
from airobas.blocks_hub.mip_blocks_lib.commons.parameters import (
    ParamsBoundComputation,
    ParamsBoundComputationEnum,
)
from airobas.blocks_hub.mip_blocks_lib.commons.utilities.evaluate import (
    evaluate,
    random_input,
)
from airobas.blocks_hub.mip_blocks_lib.linear.verification_formula.verification_constraints import (
    get_output_constrs,
)
from airobas.verif_pipeline import (
    BlockVerif,
    BlockVerifOutput,
    DataContainer,
    ProblemContainer,
    StatusVerif,
)
from gurobi_ml import add_predictor_constr

logger = logging.getLogger(__name__)


class GMLBrick(BlockVerif):
    def __init__(
        self,
        problem_container: ProblemContainer,
        data_container: DataContainer,
        **kwargs,
    ):
        super().__init__(problem_container=problem_container, data_container=data_container)
        self.model = problem_container.model
        self.lp_model: gp.Model = None
        self.x = None
        self.y = None
        self.constraint_nn = None
        self.neural_net = NeuralNetwork()
        self.options = kwargs

    def init_gml_model(self, x_min: np.ndarray, x_max: np.ndarray, y_min: np.ndarray):
        do_bounds_computation = self.options.get("do_bounds_computation", True)
        do_warm_start = self.options.get("do_warm_start", False)
        # Recompute bounds using interval arithmetic or decomon.
        spec = InputBoundsNeuralNetwork(input_lower_bounds=x_min, input_upper_bounds=x_max)
        if do_bounds_computation or do_warm_start:
            params_bound = ParamsBoundComputation(ParamsBoundComputationEnum.SIA)
            if self.neural_net.input is None:
                self.neural_net.parse_keras(self.model, spec=spec)
                if do_bounds_computation:
                    create_object_bounds(params_bound, self.neural_net).update_bounds_net(self.neural_net)
            else:
                self.neural_net = self.neural_net.clone(spec=spec)
                if do_bounds_computation:
                    create_object_bounds(params_bound, self.neural_net).update_bounds_net(self.neural_net)
        m = gp.Model()
        m.setParam("OutputFlag", 0)  # remove huge log by default
        if "OutputFlag" in self.options:
            m.setParam("OutputFlag", self.options["OutputFlag"])
        x: gp.MVar = m.addMVar(x_min.shape, lb=x_min, ub=x_max, name="x")
        y: gp.MVar = m.addMVar(y_min.shape, lb=-gp.GRB.INFINITY, name="y")
        self.constraint_nn = add_predictor_constr(m, self.model, x, y)
        self.lp_model = m
        self.lp_model.setParam("Heuristics", 0.2)
        if do_bounds_computation:
            for i in range(len(self.constraint_nn._layers)):
                output = self.constraint_nn._layers[i].output
                mixing = getattr(self.constraint_nn._layers[i], "mixing", None)
                layer = self.neural_net.layers[i + 1]
                for k in range(output.shape[1]):
                    output[0, k].LB = layer.bounds["out"]["l"][k]
                    output[0, k].UB = layer.bounds["out"]["u"][k]
                    if mixing is not None:
                        mixing[0, k].LB = layer.bounds["in"]["l"][k]
                        mixing[0, k].UB = layer.bounds["in"]["u"][k]
        self.x = x.tolist()
        self.y = y.tolist()
        if do_warm_start:
            self.do_warm_start()

    def do_warm_start(self, ri: Optional[np.ndarray] = None):
        if ri is None:
            ri = random_input(self.neural_net)
        evaluates = evaluate(input_array=ri, neural_net=self.neural_net)
        for k in range(len(self.x)):
            self.x[k].Start = evaluates[0][0][k]
        for j in range(1, len(evaluates)):
            output = self.constraint_nn._layers[j - 1].output
            mixing = getattr(self.constraint_nn._layers[j - 1], "mixing", None)
            if mixing is not None:
                mixing.Start = evaluates[j][0][None, :]
            output.Start = evaluates[j][1][None, :]

    def method_mip(self, x_min, x_max, y_min, y_max, **kwargs):
        nb_points = x_min.shape[0]
        output = BlockVerifOutput(
            status=np.empty(nb_points, dtype=StatusVerif),
            inputs=[None for i in range(nb_points)],
            outputs=[None for i in range(nb_points)],
            build_time=0,
            init_time_per_sample=np.empty(nb_points, dtype=float),
            verif_time_per_sample=np.empty(nb_points, dtype=float),
        )
        nb = 0
        for index in range(nb_points):
            nb += 1
            t1 = time.perf_counter()
            self.init_gml_model(x_min=x_min[index], x_max=x_max[index], y_min=y_min[index])
            formula = NAryConjFormula(
                [
                    VarConstConstraint(StateCoordinate(i), LT, y_max[index, i] + 10**-6)
                    for i in range(y_max.shape[1])
                    if self.neural_net.layers[-1].bounds["out"]["u"][i] > y_max[index, i]
                ]
                + [
                    VarConstConstraint(StateCoordinate(i), GT, y_min[index, i] - 10**-6)
                    for i in range(y_max.shape[1])
                    if self.neural_net.layers[-1].bounds["out"]["l"][i] < y_min[index, i]
                ]
            )
            if len(formula.clauses) == 0:
                t2 = time.perf_counter()
                output.status[index] = StatusVerif.VERIFIED
                output.init_time_per_sample[index] = t2 - t1
                output.verif_time_per_sample[index] = 0
            else:
                # Adding the output constraint to the model.
                get_output_constrs(formula, model=self.lp_model, output_vars=self.y, negate=True)
                t2 = time.perf_counter()
                self.lp_model.optimize()
                t3 = time.perf_counter()
                if self.lp_model.Status == gp.GRB.INFEASIBLE:
                    output.status[index] = StatusVerif.VERIFIED
                elif self.lp_model.ObjVal >= 0.0:
                    output.status[index] = StatusVerif.VIOLATED
                    counter_example = [x.X for x in self.x]
                    prediction = self.problem_container.model.predict([counter_example], verbose=0)[0]
                    output.inputs[index] = np.array(counter_example)
                    output.outputs[index] = prediction
                elif self.lp_model.Status in {gp.GRB.TIME_LIMIT, gp.GRB.INTERRUPTED}:
                    output.status[index] = StatusVerif.TIMEOUT
                output.init_time_per_sample[index] = t2 - t1
                output.verif_time_per_sample[index] = t3 - t2
            logger.info(f"Current sat (%) {np.sum(output.status == StatusVerif.VERIFIED) / nb_points * 100}")
            logger.info(f"Current unsat (%) {np.sum(output.status == StatusVerif.VIOLATED) / nb_points * 100}")

        return output

    def verif(self, indexes: np.ndarray) -> BlockVerifOutput:
        return self.method_mip(
            x_min=self.data_container.lbound_input_points[indexes, :],
            x_max=self.data_container.ubound_input_points[indexes, :],
            y_min=self.data_container.lbound_output_points[indexes, :],
            y_max=self.data_container.ubound_output_points[indexes, :],
        )

    @staticmethod
    def get_name() -> str:
        return "gurobi-machine-learning"
