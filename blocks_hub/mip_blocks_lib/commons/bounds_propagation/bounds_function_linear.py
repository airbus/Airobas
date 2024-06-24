from typing import Optional

import numpy as np

from blocks_hub.mip_blocks_lib.commons.layers import Layer
from blocks_hub.mip_blocks_lib.commons.linearfunctions import LinearFunctions
from blocks_hub.mip_blocks_lib.commons.parameters import Bounds


def compute_bounds(
    current_layer: Layer,
    previous_layer: Layer,
    method: Bounds,
    runtime=False,
    binary_vars: Optional[np.ndarray] = None,
    input_bounds: Optional[np.ndarray] = None,
):
    if method == Bounds.INT_ARITHMETIC:
        compute_bounds_ia(current_layer, previous_layer)
    elif method == Bounds.SYMBOLIC_INT_ARITHMETIC:
        if input_bounds is None:
            raise Exception(
                """Missing input-bounds parameter for
                            symbolic-based calculation of
                            bounds"""
            )
        else:
            compute_bounds_sia(current_layer, previous_layer, input_bounds)
    compute_error(current_layer, previous_layer)


def compute_bounds_sia(current_layer, previous_layer, input_bounds):
    weights_plus = np.maximum(
        current_layer.weights, np.zeros(current_layer.weights.shape)
    )
    weights_minus = np.minimum(
        current_layer.weights, np.zeros(current_layer.weights.shape)
    )

    # get coefficients for the bound equations
    p_l_coeffs = previous_layer.bound_equations["out"]["l"].matrix
    p_u_coeffs = previous_layer.bound_equations["out"]["u"].matrix
    l_coeffs = weights_plus.dot(p_l_coeffs) + weights_minus.dot(p_u_coeffs)
    u_coeffs = weights_plus.dot(p_u_coeffs) + weights_minus.dot(p_l_coeffs)

    # get constants for the bound equations
    p_l_const = previous_layer.bound_equations["out"]["l"].offset
    p_u_const = previous_layer.bound_equations["out"]["u"].offset
    l_const = (
        weights_plus.dot(p_l_const) + weights_minus.dot(p_u_const) + current_layer.bias
    )
    u_const = (
        weights_plus.dot(p_u_const) + weights_minus.dot(p_l_const) + current_layer.bias
    )

    # set bound equations
    current_layer.bound_equations["out"]["l"] = LinearFunctions(l_coeffs, l_const)
    current_layer.bound_equations["out"]["u"] = LinearFunctions(u_coeffs, u_const)

    # set concrete output bounds
    current_layer.bounds["out"]["l"] = current_layer.bound_equations["out"][
        "l"
    ].compute_min_values(input_bounds)
    current_layer.bounds["out"]["u"] = current_layer.bound_equations["out"][
        "u"
    ].compute_max_values(input_bounds)

    current_layer.bounds["in"]["l"] = current_layer.bounds["out"]["l"]
    current_layer.bounds["in"]["u"] = current_layer.bounds["out"]["u"]


def compute_bounds_ia(current_layer: Layer, previous_layer: Layer):
    weights_plus = np.maximum(
        current_layer.weights, np.zeros(current_layer.weights.shape)
    )
    weights_minus = np.minimum(
        current_layer.weights, np.zeros(current_layer.weights.shape)
    )

    current_layer.bounds["out"]["l"] = np.array(
        [
            weights_plus[i].dot(previous_layer.bounds["out"]["l"])
            + weights_minus[i].dot(previous_layer.bounds["out"]["u"])
            + current_layer.bias[i]
            for i in range(current_layer.output_shape)
        ]
    )

    current_layer.bounds["out"]["u"] = np.array(
        [
            weights_plus[i].dot(previous_layer.bounds["out"]["u"])
            + weights_minus[i].dot(previous_layer.bounds["out"]["l"])
            + current_layer.bias[i]
            for i in range(current_layer.output_shape)
        ]
    )


def compute_error(current_layer, previous_layer):
    current_layer.error["in"] = current_layer.weights.dot(previous_layer.error["out"])
    current_layer.error["out"] = current_layer.weights.dot(previous_layer.error["out"])
