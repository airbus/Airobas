# from typing import Optional
# from gurobipy import GRB
#
# from blocks_hub.mip_blocks_lib.commons.bounds_propagation.bounds_computation_decomon import BoundsComputationDecomon
# from blocks_hub.mip_blocks_lib.commons.bounds_propagation.linear_propagation import compute_bounds_linear_propagation
# from blocks_hub.mip_blocks_lib.commons.layers import Relu, Linear
# from blocks_hub.mip_blocks_lib.commons.neural_network import NeuralNetwork, \
#     neural_network_to_keras
# from blocks_hub.mip_blocks_lib.commons.parameters import Bounds
# from blocks_hub.mip_blocks_lib.linear.linear_model import LinearModel
# import logging
# from blocks_hub.mip_blocks_lib.commons.bounds_propagation.bounds_computation_interface import BoundComputation
# logger = logging.getLogger(__file__)
#
#
# class BoundComputationSolverMilp(BoundComputation):
#     def update_bounds(self):
#         self.optimize_bounds(neural_network=self.neural_network,
#                              time_limit_per_neuron=self.time_limit_per_neuron,
#                              nb_layer_backward=self.nb_layer_backward,
#                              verbose=self.debug_logs)
#
#     def __init__(self, neural_network: NeuralNetwork, time_limit_per_neuron: float = 3.,
#                  nb_layer_backward: Optional[int] = None,
#                  debug_logs: bool = True, **kwargs):
#         super().__init__(neural_network)
#         self.time_limit_per_neuron = time_limit_per_neuron
#         self.nb_layer_backward = nb_layer_backward
#         self.debug_logs = debug_logs
#
#     def callbacks(self):
#         def callback_relu_lb_preactivation(model, where):
#             if where == GRB.Callback.MIPNODE:
#                 status = model.cbGet(GRB.Callback.MIPNODE_STATUS)
#                 if status == GRB.OPTIMAL:
#                     obj = model.cbGet(GRB.Callback.MIPNODE_OBJBND)
#                     if obj >= 0:
#                         print("Finished")
#                         model.terminate()
#
#     def optimize_bounds(
#         self,
#         neural_network: NeuralNetwork,
#         time_limit_per_neuron: float,
#         nb_layer_backward: Optional[int] = None,
#         verbose: bool = False,
#     ):
#         b = BoundsComputationDecomon(neural_network=self.neural_network)
#         b.update_bounds()
#         # compute_bounds_linear_propagation(
#         #      lmodel=self.neural_network,
#         #      runtime=False,
#         #      binary_vars=None,
#         #      start=-1,
#         #      end=-1,
#         #      method=Bounds.SYMBOLIC_INT_ARITHMETIC,
#         # )
#         number_layer = len(neural_network.layers)
#         for i_layer in range(number_layer):
#             print("Layer ", i_layer)
#             if i_layer <= 1:
#                 continue
#             keys = neural_network.get_keys_of_layer(i_layer)
#             linear_model = LinearModel(neural_network=neural_network)
#             linear_model.init_model(
#                 first_layer=0
#                 if nb_layer_backward is None
#                 else max(i_layer - nb_layer_backward, 0),
#                 final_layer=i_layer,
#                 relax_relu_triangle=True,
#                 model_relu=True,
#                 use_big_m=True,
#                 use_max=True,
#                 use_indicator=True,
#                 create_indicator_var=True,
#                 add_linear_ub_and_lb=False,
#                 solve=False,
#             )
#             for key in keys:
#                 print("Computing ...", key, " out of ", len(keys))
#                 linear_model.model.setParam("OutputFlag", 0)
#                 for preactivation in [True, False]:
#                     key_activ = (
#                         "x_preactivation" if preactivation else "x_postactivation"
#                     )
#                     maximisation = not preactivation
#                     linear_model.model.setObjective(
#                         linear_model.variables[key_activ][key],
#                         sense=GRB.MAXIMIZE if maximisation else GRB.MINIMIZE,
#                     )
#                     linear_model.model.setParam("TimeLimit", time_limit_per_neuron)
#                     linear_model.model.setParam("Heuristics", 0.2)
#                     linear_model.model.optimize()
#                     a = linear_model.model.ObjBound
#                     if a == float("inf") or a == -float("inf"):
#                         print("--")
#                         continue
#                     if maximisation:
#                         if verbose:
#                             print(
#                                 "Prev, out/u",
#                                 neural_network.layers[i_layer].bounds["out"]["u"][
#                                     key[1]
#                                 ],
#                                 "New",
#                                 a,
#                             )
#                         if abs(a) < 10e-10:
#                             a = 0
#                         prev = neural_network.layers[i_layer].bounds["out"]["u"][key[1]]
#                         if key[0] > 0 and isinstance(
#                             self.neural_network.layers[key[0]], Relu
#                         ):
#                             if a < prev:
#                                 neural_network.layers[i_layer].bounds["out"]["u"][
#                                     key[1]
#                                 ] = min(
#                                     neural_network.layers[i_layer].bounds["out"]["u"][
#                                         key[1]
#                                     ],
#                                     a,
#                                 )
#                                 linear_model.variables["x_postactivation"][key].UB = \
#                                     neural_network.layers[i_layer].bounds["out"]["u"][
#                                         key[1]
#                                     ]
#                         if (
#                             prev > 0
#                             and a == 0
#                             and key[0] > 0
#                             and isinstance(self.neural_network.layers[key[0]], Relu)
#                         ):
#                             print(key, "Becomes inactive !")
#                         neural_network.layers[i_layer].bounds["out"]["u"][key[1]] = min(
#                             neural_network.layers[i_layer].bounds["out"]["u"][key[1]], a
#                         )
#                         linear_model.variables["x_postactivation"][key].UB = \
#                             neural_network.layers[i_layer].bounds["out"]["u"][
#                                 key[1]
#                             ]
#                         if isinstance(self.neural_network.layers[key[0]], Linear):
#                             neural_network.layers[i_layer].bounds["in"]["u"][key[1]] = neural_network.layers[i_layer].bounds["out"]["u"][key[1]]
#                             neural_network.layers[i_layer].bounds["out"]["l"][key[1]] = neural_network.layers[i_layer].bounds["in"]["l"][key[1]]
#
#                     else:
#                         if verbose:
#                             print(
#                                 "Prev, in/l",
#                                 neural_network.layers[i_layer].bounds["in"]["l"][
#                                     key[1]
#                                 ],
#                                 "New",
#                                 a,
#                             )
#                         if abs(a) <= 10e-10:
#                             a = 0
#                         if (
#                             neural_network.layers[i_layer].bounds["in"]["u"][key[1]]
#                             > 0
#                             > neural_network.layers[i_layer].bounds["in"]["l"][key[1]]
#                             and a >= 0
#                             and key[0] > 0
#                             and isinstance(self.neural_network.layers[key[0]], Relu)
#                         ):
#                             print(key, "Becomes active !")
#                             neural_network.layers[i_layer].bounds["in"]["l"][
#                                 key[1]
#                             ] = max(
#                                 neural_network.layers[i_layer].bounds["in"]["l"][
#                                     key[1]
#                                 ],
#                                 a,
#                             )
#                             linear_model.variables["x_preactivation"][key].LB = \
#                             neural_network.layers[i_layer].bounds["in"]["l"][key[1]]
#
#                         if isinstance(self.neural_network.layers[key[0]], Relu):
#                             neural_network.layers[i_layer].bounds["out"]["l"][
#                                 key[1]
#                             ] = max(
#                                 neural_network.layers[i_layer].bounds["out"]["l"][
#                                     key[1]
#                                 ],
#                                 a,
#                             )
#                             linear_model.variables["x_postactivation"][key].LB = (
#                                 neural_network.layers[i_layer].bounds["out"]["l"][
#                                 key[1]
#                             ])
#                         if isinstance(self.neural_network.layers[key[0]], Linear):
#                             neural_network.layers[i_layer].bounds["out"]["l"][
#                                 key[1]
#                             ] = max(
#                                 neural_network.layers[i_layer].bounds["out"]["l"][
#                                     key[1]
#                                 ],
#                                 a,
#                             )
#                             linear_model.variables["x_postactivation"][key].LB = neural_network.layers[i_layer].bounds["out"]["l"][
#                                 key[1]
#                             ]
#                         neural_network.layers[i_layer].bounds["in"]["l"][key[1]] = max(
#                             neural_network.layers[i_layer].bounds["in"]["l"][key[1]], a
#                         )
#                         linear_model.variables["x_preactivation"][key].LB = neural_network.layers[i_layer].bounds["in"]["l"][key[1]]
#
