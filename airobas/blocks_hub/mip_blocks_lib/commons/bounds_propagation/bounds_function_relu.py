from typing import Dict, Optional

import numpy as np
from airobas.blocks_hub.mip_blocks_lib.commons.layers import Layer
from airobas.blocks_hub.mip_blocks_lib.commons.linearfunctions import LinearFunctions
from airobas.blocks_hub.mip_blocks_lib.commons.parameters import Bounds


def compute_bounds(
    current_layer: Layer,
    previous_layer: Layer,
    method: Bounds,
    runtime=False,
    binary_vars: Optional[np.ndarray] = None,
    input_bounds: Optional[np.ndarray] = None,
):
    if binary_vars is None:
        binary_vars = []
    if method == Bounds.INT_ARITHMETIC:
        if runtime:
            if len(binary_vars) == 0:
                raise Exception(
                    """Missing binary_vars parameter for runtime
                calculation of bounds"""
                )
            else:
                compute_runtime_bounds_interval_arithmetic(current_layer, previous_layer, binary_vars)
        else:
            compute_bounds_interval_arithmetic(current_layer=current_layer, previous_layer=previous_layer)
    elif method == Bounds.SYMBOLIC_INT_ARITHMETIC:
        if len(input_bounds) == 0:
            raise Exception(
                """Missing input-bounds parameter for
                            symbolic-based calculation of
                            bounds"""
            )
        else:
            if runtime:
                if len(binary_vars) == 0:
                    raise Exception(
                        """Missing binary_vars parameter for
                    runtime calculation of bounds"""
                    )
                else:
                    compute_runtime_bounds_sia(
                        current_layer=current_layer,
                        previous_layer=previous_layer,
                        input_bounds=input_bounds,
                        binary_vars=binary_vars,
                    )
            else:
                compute_bounds_sia(
                    current_layer=current_layer,
                    previous_layer=previous_layer,
                    input_bounds=input_bounds,
                )

    compute_error(current_layer, previous_layer)


def compute_error(current_layer, previous_layer):
    current_layer.error["in"] = current_layer.weights.dot(previous_layer.error["out"])
    current_layer.error["out"] = np.empty(current_layer.output_shape)
    for i in range(current_layer.output_shape):
        ub = current_layer.bounds["in"]["u"][i]
        lb = current_layer.bounds["in"]["l"][i]

        if current_layer.is_fixed(i):
            current_layer.error["out"][i] = 0
        else:
            act_area = (pow(ub, 2) / 2) - (pow(ub, 3) / (2 * (ub - lb)))
            inact_area = (pow(lb, 2) * ub) / (2 * (ub - lb))
            area1 = act_area + inact_area
            area2 = pow(ub, 2) / 2
            whole_area = (ub * (ub - lb)) - (lb * ub)

            if area2 < area1:
                current_layer.error["out"][i] = area2 / whole_area
            else:
                current_layer.error["out"][i] = area1 / whole_area


# Constraint propagation
def compute_bounds_interval_arithmetic(current_layer: Layer, previous_layer: Layer):
    current_layer.bounds["in"] = compute_in_interval_arithmetic(
        current_layer=current_layer,
        input_lb=previous_layer.bounds["out"]["l"],
        input_ub=previous_layer.bounds["out"]["u"],
    )
    current_layer.bounds["out"] = compute_out_interval_arithmetic(
        current_layer=current_layer, in_bounds=current_layer.bounds["in"]
    )


def compute_in_interval_arithmetic(current_layer: Layer, input_lb: np.ndarray, input_ub: np.ndarray):
    weights_plus = np.maximum(current_layer.weights, np.zeros(current_layer.weights.shape))
    weights_minus = np.minimum(current_layer.weights, np.zeros(current_layer.weights.shape))

    lower = np.array(
        [
            weights_plus[i].dot(input_lb) + weights_minus[i].dot(input_ub) + current_layer.bias[i]
            for i in range(current_layer.output_shape)
        ]
    )

    upper = np.array(
        [
            weights_plus[i].dot(input_ub) + weights_minus[i].dot(input_lb) + current_layer.bias[i]
            for i in range(current_layer.output_shape)
        ]
    )
    return {"l": lower, "u": upper}


def compute_out_interval_arithmetic(current_layer: Layer, in_bounds: Dict[str, np.ndarray]):
    lower = np.maximum(in_bounds["l"], np.zeros(current_layer.output_shape))
    upper = np.maximum(in_bounds["u"], np.zeros(current_layer.output_shape))
    return {"l": lower, "u": upper}


def compute_runtime_bounds_interval_arithmetic(current_layer: Layer, previous_layer: Layer, binary_vars):
    """
    Bounds  - Interval Arithmetic - Runtime computation
    """
    current_layer.rnt_bounds["in"] = compute_in_interval_arithmetic(
        current_layer=current_layer,
        input_lb=previous_layer.rnt_bounds["out"]["l"],
        input_ub=previous_layer.rnt_bounds["out"]["u"],
    )
    current_layer.rnt_bounds["out"] = compute_runtime_out_interval_arithmetic(
        current_layer, current_layer.rnt_bounds["in"], binary_vars
    )


def compute_runtime_out_interval_arithmetic(current_layer: Layer, in_bounds: Dict[str, np.ndarray], binary_vars):
    bounds = compute_out_interval_arithmetic(current_layer, in_bounds)
    bounds["l"][(binary_vars == 0)] = 0
    bounds["u"][(binary_vars == 0)] = 0
    return bounds


#
# Bounds - Symbolic Interval Arithmetic - Pre-computation
#
def compute_bounds_sia(current_layer: Layer, previous_layer: Layer, input_bounds: np.ndarray):
    # set input bounds equations
    current_layer.bound_equations["in"] = compute_in_bound_eqs(
        current_layer=current_layer,
        p_low_eq=previous_layer.bound_equations["out"]["l"],
        p_up_eq=previous_layer.bound_equations["out"]["u"],
    )
    # set concrete input bounds
    current_layer.bounds["in"]["l"] = current_layer.bound_equations["in"]["l"].compute_min_values(input_bounds)
    current_layer.bounds["in"]["u"] = current_layer.bound_equations["in"]["u"].compute_max_values(input_bounds)

    # set output bounds equatuons
    current_layer.bound_equations["out"] = compute_out_bound_eqs(current_layer.bound_equations["in"], input_bounds)
    # set concrete output bounds
    current_layer.bounds["out"]["l"] = current_layer.bound_equations["out"]["l"].compute_min_values(input_bounds)
    current_layer.bounds["out"]["u"] = current_layer.bound_equations["out"]["u"].compute_max_values(input_bounds)
    # make sure the bounds are not below zero
    current_layer.bounds["out"]["l"] = np.maximum(
        current_layer.bounds["out"]["l"], np.zeros(current_layer.output_shape)
    )
    current_layer.bounds["out"]["u"] = np.maximum(
        current_layer.bounds["out"]["u"], np.zeros(current_layer.output_shape)
    )


def compute_in_bound_eqs(current_layer: Layer, p_low_eq: LinearFunctions, p_up_eq: LinearFunctions):
    weights_plus = np.maximum(current_layer.weights, np.zeros(current_layer.weights.shape))
    weights_minus = np.minimum(current_layer.weights, np.zeros(current_layer.weights.shape))

    # get coefficients for the input bound equations
    p_l_coeffs = p_low_eq.matrix
    p_u_coeffs = p_up_eq.matrix
    l_coeffs = weights_plus.dot(p_l_coeffs) + weights_minus.dot(p_u_coeffs)
    u_coeffs = weights_plus.dot(p_u_coeffs) + weights_minus.dot(p_l_coeffs)

    # get constants for the input bound equations
    p_l_const = p_low_eq.offset
    p_u_const = p_up_eq.offset
    l_const = weights_plus.dot(p_l_const) + weights_minus.dot(p_u_const) + current_layer.bias
    u_const = weights_plus.dot(p_u_const) + weights_minus.dot(p_l_const) + current_layer.bias
    # return input bound equations
    return {
        "l": LinearFunctions(l_coeffs, l_const),
        "u": LinearFunctions(u_coeffs, u_const),
    }


def compute_out_bound_eqs(in_eqs: Dict[str, LinearFunctions], input_bounds: np.ndarray):
    # return out bound equations
    return {
        "l": in_eqs["l"].get_lower_relu_relax(input_bounds),
        "u": in_eqs["u"].get_upper_relu_relax(input_bounds),
    }


def compute_runtime_bounds_sia(current_layer: Layer, previous_layer: Layer, input_bounds: np.ndarray, binary_vars):
    # set input runtime bounds equations
    current_layer.rnt_bound_equations["in"] = compute_runtime_in_bound_eqs(
        current_layer,
        previous_layer.rnt_bound_equations["out"]["l"],
        previous_layer.rnt_bound_equations["out"]["u"],
    )
    # set concrete input runtime bounds
    current_layer.rnt_bounds["in"]["l"] = current_layer.rnt_bound_equations["in"]["l"].compute_min_values(input_bounds)
    current_layer.rnt_bounds["in"]["u"] = current_layer.rnt_bound_equations["in"]["u"].compute_max_values(input_bounds)

    # set output bounds equations
    current_layer.rnt_bound_equations["out"] = compute_runtime_out_bound_eqs(
        current_layer.rnt_bound_equations["in"], input_bounds, binary_vars
    )
    # set concrete output bounds
    current_layer.rnt_bounds["out"]["l"] = current_layer.rnt_bound_equations["out"]["l"].compute_min_values(
        input_bounds
    )
    current_layer.rnt_bounds["out"]["u"] = current_layer.rnt_bound_equations["out"]["u"].compute_max_values(
        input_bounds
    )
    # make sure the bounds are not below zero
    current_layer.rnt_bounds["out"]["l"] = np.maximum(
        current_layer.rnt_bounds["out"]["l"], np.zeros(current_layer.output_shape)
    )
    current_layer.rnt_bounds["out"]["u"] = np.maximum(
        current_layer.rnt_bounds["out"]["u"], np.zeros(current_layer.output_shape)
    )


def compute_runtime_in_bound_eqs(current_layer: Layer, p_low_eq: LinearFunctions, p_up_eq: LinearFunctions):
    return compute_in_bound_eqs(current_layer, p_low_eq, p_up_eq)


def compute_runtime_out_bound_eqs(in_eqs: Dict[str, LinearFunctions], input_bounds: np.ndarray, binary_vars):
    # set out bound equations
    eqs = compute_out_bound_eqs(in_eqs, input_bounds)
    # set bound functions to zero for inactive nodes
    eqs["l"].matrix[(binary_vars == 0), :] = 0
    eqs["l"].offset[(binary_vars == 0)] = 0
    eqs["u"].matrix[(binary_vars == 0), :] = 0
    eqs["u"].offset[(binary_vars == 0)] = 0
    # set bound functions to input ones for active nodes
    eqs["l"].matrix[(binary_vars == 1), :] = in_eqs["l"].matrix[(binary_vars == 1), :]
    eqs["l"].offset[(binary_vars == 1)] = in_eqs["l"].offset[(binary_vars == 1)]
    eqs["u"].matrix[(binary_vars == 1), :] = in_eqs["u"].matrix[(binary_vars == 1), :]
    eqs["u"].offset[(binary_vars == 1)] = in_eqs["u"].offset[(binary_vars == 1)]

    return eqs
