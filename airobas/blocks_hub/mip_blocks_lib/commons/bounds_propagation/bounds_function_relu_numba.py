from typing import Dict, Optional

import numpy as np
from numba import njit

from airobas.blocks_hub.mip_blocks_lib.commons.layers import Layer, Relu
from airobas.blocks_hub.mip_blocks_lib.commons.linearfunctions import (
    LinearFunctions,
    compute_max_values_numba,
    compute_min_values_numba,
    compute_upper_numba,
    get_lower_relu_relax_numba,
    get_upper_relu_relax_numba,
)
from airobas.blocks_hub.mip_blocks_lib.commons.parameters import Bounds


def compute_bounds_numba(
    current_layer: Layer,
    previous_layer: Layer,
    method: Bounds,
    runtime=False,
    binary_vars: Optional[np.ndarray] = None,
    input_lower: Optional[np.ndarray] = None,
    input_upper: Optional[np.ndarray] = None,
):
    if binary_vars is None:
        binary_vars = []
    is_relu = isinstance(current_layer, Relu)
    if method == Bounds.INT_ARITHMETIC:
        if runtime:
            if len(binary_vars) == 0:
                raise Exception(
                    """Missing binary_vars parameter for runtime
                calculation of bounds"""
                )
            else:
                compute_runtime_bounds_interval_arithmetic(
                    current_layer=current_layer,
                    previous_layer=previous_layer,
                    binary_vars=binary_vars,
                    is_relu=is_relu,
                )
        else:
            compute_bounds_interval_arithmetic(
                current_layer=current_layer,
                previous_layer=previous_layer,
                is_relu=is_relu,
            )
    elif method == Bounds.SYMBOLIC_INT_ARITHMETIC:
        if input_upper is None or input_lower is None:
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
                        input_lower=input_lower,
                        input_upper=input_upper,
                        binary_vars=binary_vars,
                        is_relu=is_relu,
                    )
            else:
                compute_bounds_sia(
                    current_layer=current_layer,
                    previous_layer=previous_layer,
                    input_lower=input_lower,
                    input_upper=input_upper,
                    is_relu=is_relu,
                )
    compute_error(current_layer, previous_layer)


def compute_error(current_layer: Layer, previous_layer: Layer):
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
def compute_bounds_interval_arithmetic(
    current_layer: Layer, previous_layer: Layer, is_relu: bool
):
    current_layer.bounds["in"] = compute_in_interval_arithmetic(
        weights_current_layer_minus=current_layer.weights_minus,
        weights_current_layer_plus=current_layer.weights_plus,
        bias_current_mayer=current_layer.bias,
        output_shape=current_layer.output_shape,
        input_lb=previous_layer.bounds["out"]["l"],
        input_ub=previous_layer.bounds["out"]["u"],
    )
    if is_relu:
        current_layer.bounds["out"] = compute_out_interval_arithmetic(
            output_shape=current_layer.output_shape,
            lower_bounds=current_layer.bounds["in"]["l"],
            upper_bounds=current_layer.bounds["in"]["u"],
        )
    else:
        current_layer.bounds["out"] = current_layer.bounds["in"]


@njit
def compute_in_interval_arithmetic(
    weights_current_layer_plus: np.ndarray,
    weights_current_layer_minus: np.ndarray,
    bias_current_mayer: np.ndarray,
    output_shape: int,
    input_lb: np.ndarray,
    input_ub: np.ndarray,
):
    weights_plus = weights_current_layer_plus
    weights_minus = weights_current_layer_minus

    lower = np.array(
        [
            weights_plus[i].dot(input_lb)
            + weights_minus[i].dot(input_ub)
            + bias_current_mayer[i]
            for i in range(output_shape)
        ]
    )

    upper = np.array(
        [
            weights_plus[i].dot(input_ub)
            + weights_minus[i].dot(input_lb)
            + bias_current_mayer[i]
            for i in range(output_shape)
        ]
    )
    return {"l": lower, "u": upper}


@njit
def compute_out_interval_arithmetic(
    output_shape: int, lower_bounds: np.ndarray, upper_bounds: np.ndarray
):
    lower = np.maximum(lower_bounds, 0)
    upper = np.maximum(upper_bounds, 0)
    return {"l": lower, "u": upper}


def compute_runtime_bounds_interval_arithmetic(
    current_layer: Layer, previous_layer: Layer, binary_vars: np.ndarray, is_relu: bool
):
    """
    Bounds  - Interval Arithmetic - Runtime computation
    """
    current_layer.rnt_bounds["in"] = compute_in_interval_arithmetic(
        weights_current_layer_minus=current_layer.weights_minus,
        weights_current_layer_plus=current_layer.weights_plus,
        bias_current_mayer=current_layer.bias,
        output_shape=current_layer.output_shape,
        input_lb=previous_layer.rnt_bounds["out"]["l"],
        input_ub=previous_layer.rnt_bounds["out"]["u"],
    )
    current_layer.rnt_bounds["out"] = compute_runtime_out_interval_arithmetic(
        output_shape=current_layer.output_shape,
        lower_bounds=current_layer.rnt_bounds["in"]["l"],
        upper_bounds=current_layer.rnt_bounds["in"]["u"],
        binary_vars=binary_vars,
        is_relu=is_relu,
    )


@njit
def compute_runtime_out_interval_arithmetic(
    output_shape: int,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    binary_vars: np.ndarray,
    is_relu: bool,
):
    bounds = compute_out_interval_arithmetic(
        output_shape=output_shape, lower_bounds=lower_bounds, upper_bounds=upper_bounds
    )
    if is_relu:
        bounds["l"][(binary_vars == 0)] = 0
        bounds["u"][(binary_vars == 0)] = 0
    return bounds


#
# Bounds - Symbolic Interval Arithmetic - Pre-computation
#
def compute_bounds_sia(
    current_layer: Layer,
    previous_layer: Layer,
    input_lower: np.ndarray,
    input_upper: np.ndarray,
    is_relu: bool,
):
    # set input bounds equations
    l_coeffs, l_const, u_coeffs, u_const = compute_in_bound_eqs(
        current_layer_weights_minus=current_layer.weights_minus,
        current_layer_weights_plus=current_layer.weights_plus,
        current_layer_bias=current_layer.bias,
        previous_layer_lower_equation_matrix=previous_layer.bound_equations["out"][
            "l"
        ].matrix,
        previous_layer_lower_equation_offset=previous_layer.bound_equations["out"][
            "l"
        ].offset,
        previous_layer_upper_equation_matrix=previous_layer.bound_equations["out"][
            "u"
        ].matrix,
        previous_layer_upper_equation_offset=previous_layer.bound_equations["out"][
            "u"
        ].offset,
    )
    current_layer.bound_equations["in"] = {
        "l": LinearFunctions(l_coeffs, l_const),
        "u": LinearFunctions(u_coeffs, u_const),
    }
    # set concrete input bounds
    current_layer.bounds["in"]["l"] = compute_min_values_numba(
        matrix=current_layer.bound_equations["in"]["l"].matrix,
        offset=current_layer.bound_equations["in"]["l"].offset,
        input_lower=input_lower,
        input_upper=input_upper,
    )
    current_layer.bounds["in"]["u"] = compute_max_values_numba(
        matrix=current_layer.bound_equations["in"]["u"].matrix,
        offset=current_layer.bound_equations["in"]["u"].offset,
        input_lower=input_lower,
        input_upper=input_upper,
    )

    # set output bounds equatuons
    if is_relu:
        l_coeffs_o, l_const_o, u_coeffs_o, u_const_o = compute_out_bound_eqs(
            equation_matrix_l=current_layer.bound_equations["in"]["l"].matrix,
            equation_offset_l=current_layer.bound_equations["in"]["l"].offset,
            equation_matrix_u=current_layer.bound_equations["in"]["u"].matrix,
            equation_offset_u=current_layer.bound_equations["in"]["u"].offset,
            size_l=current_layer.bound_equations["in"]["l"].size,
            input_lower=input_lower,
            input_upper=input_upper,
        )
        current_layer.bound_equations["out"] = {
            "l": LinearFunctions(l_coeffs_o, l_const_o),
            "u": LinearFunctions(u_coeffs_o, u_const_o),
        }
        current_layer.bounds["out"]["l"] = compute_min_values_numba(
            matrix=current_layer.bound_equations["out"]["l"].matrix,
            offset=current_layer.bound_equations["out"]["l"].offset,
            input_lower=input_lower,
            input_upper=input_upper,
        )
        current_layer.bounds["out"]["u"] = compute_max_values_numba(
            matrix=current_layer.bound_equations["out"]["u"].matrix,
            offset=current_layer.bound_equations["out"]["u"].offset,
            input_lower=input_lower,
            input_upper=input_upper,
        )
    else:
        current_layer.bound_equations["out"] = current_layer.bound_equations["in"]
        current_layer.bounds["out"]["l"] = current_layer.bounds["in"]["l"]
        current_layer.bounds["out"]["u"] = current_layer.bounds["in"]["u"]
    # set concrete output bounds

    # make sure the bounds are not below zero
    if is_relu:
        current_layer.bounds["out"]["l"] = np.maximum(
            current_layer.bounds["out"]["l"], 0
        )
        current_layer.bounds["out"]["u"] = np.maximum(
            current_layer.bounds["out"]["u"], 0
        )


@njit
def compute_in_bound_eqs(
    current_layer_weights_plus: np.ndarray,
    current_layer_weights_minus: np.ndarray,
    current_layer_bias: np.ndarray,
    previous_layer_lower_equation_matrix: np.ndarray,
    previous_layer_lower_equation_offset: np.ndarray,
    previous_layer_upper_equation_matrix: np.ndarray,
    previous_layer_upper_equation_offset: np.ndarray,
):
    weights_plus = current_layer_weights_plus
    weights_minus = current_layer_weights_minus

    # get coefficients for the input bound equations
    p_l_coeffs = previous_layer_lower_equation_matrix
    p_u_coeffs = previous_layer_upper_equation_matrix
    l_coeffs = weights_plus.dot(p_l_coeffs) + weights_minus.dot(p_u_coeffs)
    u_coeffs = weights_plus.dot(p_u_coeffs) + weights_minus.dot(p_l_coeffs)

    # get constants for the input bound equations
    p_l_const = previous_layer_lower_equation_offset
    p_u_const = previous_layer_upper_equation_offset
    l_const = (
        weights_plus.dot(p_l_const) + weights_minus.dot(p_u_const) + current_layer_bias
    )
    u_const = (
        weights_plus.dot(p_u_const) + weights_minus.dot(p_l_const) + current_layer_bias
    )
    # return input bound equations
    return l_coeffs, l_const, u_coeffs, u_const


@njit
def compute_out_bound_eqs(
    equation_matrix_l: np.ndarray,
    equation_offset_l: np.ndarray,
    size_l: int,
    equation_matrix_u: np.ndarray,
    equation_offset_u: np.ndarray,
    input_lower: np.ndarray,
    input_upper: np.ndarray,
):
    # return out bound equations
    l_coeffs, l_const = get_lower_relu_relax_numba(
        matrix=equation_matrix_l,
        offset=equation_offset_l,
        size=size_l,
        input_lower=input_lower,
        input_upper=input_upper,
    )
    u_coeffs, u_const = get_upper_relu_relax_numba(
        matrix=equation_matrix_u,
        offset=equation_offset_u,
        size=size_l,
        input_lower=input_lower,
        input_upper=input_upper,
    )
    return l_coeffs, l_const, u_coeffs, u_const


def compute_runtime_bounds_sia(
    current_layer: Layer,
    previous_layer: Layer,
    input_lower: np.ndarray,
    input_upper: np.ndarray,
    is_relu: bool,
    binary_vars: np.ndarray,
):
    # set input runtime bounds equations
    l_coeffs, l_const, u_coeffs, u_const = compute_in_bound_eqs(
        current_layer_weights_minus=current_layer.weights_minus,
        current_layer_weights_plus=current_layer.weights_plus,
        current_layer_bias=current_layer.bias,
        previous_layer_lower_equation_matrix=previous_layer.rnt_bound_equations["out"][
            "l"
        ].matrix,
        previous_layer_lower_equation_offset=previous_layer.rnt_bound_equations["out"][
            "l"
        ].offset,
        previous_layer_upper_equation_matrix=previous_layer.rnt_bound_equations["out"][
            "u"
        ].matrix,
        previous_layer_upper_equation_offset=previous_layer.rnt_bound_equations["out"][
            "u"
        ].offset,
    )
    current_layer.rnt_bound_equations["in"] = {
        "l": LinearFunctions(l_coeffs, l_const),
        "u": LinearFunctions(u_coeffs, u_const),
    }
    # set concrete input bounds
    current_layer.rnt_bounds["in"]["l"] = compute_min_values_numba(
        matrix=current_layer.rnt_bound_equations["in"]["l"].matrix,
        offset=current_layer.rnt_bound_equations["in"]["l"].offset,
        input_lower=input_lower,
        input_upper=input_upper,
    )
    current_layer.rnt_bounds["in"]["u"] = compute_max_values_numba(
        matrix=current_layer.bound_equations["in"]["u"].matrix,
        offset=current_layer.bound_equations["in"]["u"].offset,
        input_lower=input_lower,
        input_upper=input_upper,
    )

    # set output bounds equations
    if is_relu:
        l_coeffs_o, l_const_o, u_coeffs_o, u_const_o = compute_runtime_out_bound_eqs(
            equation_matrix_l=current_layer.rnt_bound_equations["in"]["l"].matrix,
            equation_offset_l=current_layer.rnt_bound_equations["in"]["l"].offset,
            equation_matrix_u=current_layer.rnt_bound_equations["in"]["u"].matrix,
            equation_offset_u=current_layer.rnt_bound_equations["in"]["u"].offset,
            size_l=current_layer.rnt_bound_equations["in"]["l"].matrix.size,
            input_lower=input_lower,
            input_upper=input_upper,
            binary_vars=binary_vars,
        )
        current_layer.rnt_bound_equations["out"] = {
            "l": LinearFunctions(l_coeffs_o, l_const_o),
            "u": LinearFunctions(u_coeffs_o, u_const_o),
        }

        # set concrete output bounds
        current_layer.rnt_bounds["out"]["l"] = compute_min_values_numba(
            matrix=current_layer.rnt_bound_equations["out"]["l"].matrix,
            offset=current_layer.rnt_bound_equations["out"]["l"].offset,
            input_lower=input_lower,
            input_upper=input_upper,
        )
        current_layer.rnt_bounds["out"]["u"] = compute_max_values_numba(
            matrix=current_layer.rnt_bound_equations["out"]["u"].matrix,
            offset=current_layer.rnt_bound_equations["out"]["u"].offset,
            input_lower=input_lower,
            input_upper=input_upper,
        )
    else:
        current_layer.rnt_bound_equations["out"] = current_layer.rnt_bound_equations[
            "in"
        ]
        # set concrete output bounds
        current_layer.rnt_bounds["out"]["l"] = current_layer.rnt_bounds["in"]["l"]
        current_layer.rnt_bounds["out"]["u"] = current_layer.rnt_bounds["in"]["l"]

    if is_relu:
        # make sure the bounds are not below zero
        current_layer.rnt_bounds["out"]["l"] = np.maximum(
            current_layer.rnt_bounds["out"]["l"], 0
        )
        current_layer.rnt_bounds["out"]["u"] = np.maximum(
            current_layer.rnt_bounds["out"]["u"], 0
        )


def compute_runtime_in_bound_eqs(
    current_layer_weights: np.ndarray,
    current_layer_bias: np.ndarray,
    previous_layer_lower_equation_matrix: np.ndarray,
    previous_layer_lower_equation_offset: np.ndarray,
    previous_layer_upper_equation_matrix: np.ndarray,
    previous_layer_upper_equation_offset: np.ndarray,
):
    return compute_in_bound_eqs(
        current_layer_weights=current_layer_weights,
        current_layer_bias=current_layer_bias,
        previous_layer_lower_equation_matrix=previous_layer_lower_equation_matrix,
        previous_layer_lower_equation_offset=previous_layer_lower_equation_offset,
        previous_layer_upper_equation_matrix=previous_layer_upper_equation_matrix,
        previous_layer_upper_equation_offset=previous_layer_upper_equation_offset,
    )


def compute_runtime_out_bound_eqs(
    equation_matrix_l: np.ndarray,
    equation_offset_l: np.ndarray,
    size_l: int,
    equation_matrix_u: np.ndarray,
    equation_offset_u: np.ndarray,
    input_lower: np.ndarray,
    input_upper: np.ndarray,
    binary_vars: np.ndarray,
):
    # set out bound equations
    l_coeffs, l_const, u_coeffs, u_const = compute_out_bound_eqs(
        equation_matrix_l=equation_matrix_l,
        equation_offset_l=equation_offset_l,
        equation_matrix_u=equation_matrix_u,
        equation_offset_u=equation_offset_u,
        size_l=size_l,
        input_lower=input_lower,
        input_upper=input_upper,
    )
    eqs = {
        "l": LinearFunctions(l_coeffs, l_const),
        "u": LinearFunctions(u_coeffs, u_const),
    }
    # set bound functions to zero for inactive nodes
    l_coeffs[(binary_vars == 0), :] = 0
    l_const[(binary_vars == 0)] = 0
    u_coeffs[(binary_vars == 0), :] = 0
    u_const[(binary_vars == 0)] = 0
    # set bound functions to input ones for active nodes
    l_coeffs[(binary_vars == 1), :] = equation_matrix_l[(binary_vars == 1), :]
    l_const[(binary_vars == 1)] = equation_offset_l[(binary_vars == 1)]
    u_coeffs[(binary_vars == 1), :] = equation_matrix_u[(binary_vars == 1), :]
    u_const[(binary_vars == 1)] = equation_offset_u[(binary_vars == 1)]

    return l_coeffs, l_const, u_coeffs, u_const
