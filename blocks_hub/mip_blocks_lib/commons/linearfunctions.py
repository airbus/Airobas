import numpy as np
from numba import njit


def compute_lower(
    weights_minus: np.ndarray,
    weights_plus: np.ndarray,
    input_lower: np.ndarray,
    input_upper: np.ndarray,
):
    return weights_plus.dot(input_lower) + weights_minus.dot(input_upper)


def compute_upper(
    weights_minus: np.ndarray,
    weights_plus: np.ndarray,
    input_lower: np.ndarray,
    input_upper: np.ndarray,
):
    return weights_plus.dot(input_upper) + weights_minus.dot(input_lower)


@njit
def compute_lower_numba(
    weights_minus: np.ndarray,
    weights_plus: np.ndarray,
    input_lower: np.ndarray,
    input_upper: np.ndarray,
):
    return weights_plus.dot(input_lower) + weights_minus.dot(input_upper)


@njit
def compute_upper_numba(
    weights_minus: np.ndarray,
    weights_plus: np.ndarray,
    input_lower: np.ndarray,
    input_upper: np.ndarray,
):
    return weights_plus.dot(input_upper) + weights_minus.dot(input_lower)


@njit
def compute_min_max_values_numba(
    matrix: np.ndarray,
    offset: np.ndarray,
    input_lower: np.ndarray,
    input_upper: np.ndarray,
):
    weights_plus = np.maximum(matrix, np.zeros(matrix.shape))
    weights_minus = np.minimum(matrix, np.zeros(matrix.shape))
    return (
        compute_lower_numba(weights_minus, weights_plus, input_lower, input_upper)
        + offset,
        compute_upper_numba(weights_minus, weights_plus, input_lower, input_upper)
        + offset,
    )


@njit
def compute_max_values_numba(
    matrix: np.ndarray,
    offset: np.ndarray,
    input_lower: np.ndarray,
    input_upper: np.ndarray,
):
    weights_plus = np.maximum(matrix, np.zeros(matrix.shape))
    weights_minus = np.minimum(matrix, np.zeros(matrix.shape))
    return (
        compute_upper_numba(weights_minus, weights_plus, input_lower, input_upper)
        + offset
    )


@njit
def compute_min_values_numba(
    matrix: np.ndarray,
    offset: np.ndarray,
    input_lower: np.ndarray,
    input_upper: np.ndarray,
):
    weights_plus = np.maximum(matrix, np.zeros(matrix.shape))
    weights_minus = np.minimum(matrix, np.zeros(matrix.shape))
    return (
        compute_lower_numba(weights_minus, weights_plus, input_lower, input_upper)
        + offset
    )


@njit
def get_lower_relu_relax_numba(
    matrix: np.ndarray,
    offset: np.ndarray,
    size: int,
    input_lower: np.ndarray,
    input_upper: np.ndarray,
):
    # get lower and upper bounds for the bound equation
    lower = compute_min_values_numba(
        matrix=matrix, offset=offset, input_lower=input_lower, input_upper=input_upper
    )
    upper = compute_max_values_numba(
        matrix=matrix, offset=offset, input_lower=input_lower, input_upper=input_upper
    )
    # matrix and offset for the relaxation
    # compute the coefficients of the linear approximation of out
    # bound equations
    for i in range(size):
        if lower[i] >= 0:
            # Active node - Propagate lower bound equation unaltered
            pass
        elif upper[i] <= 0:
            # Inactive node - Propagate the zero function
            matrix[i, :] = 0
            offset[i] = 0
        else:
            # Unstable node - Propagate linear relaxation of
            # lower bound equations
            # Approach based on over approximation areas
            lb = lower[i]
            lb2 = lb**2
            ub = upper[i]
            ub2 = ub**2
            # over-approximation area based on double
            # linear relaxation
            area1 = ((ub * lb2) - (ub2 * lb)) / (2 * (ub - lb))
            # over-approximation area based on triangle
            # relaxation
            area2 = ub**2 / 2
            # choose the least over-approximation
            if area1 < area2:
                adj = upper[i] / (upper[i] - lower[i])
                matrix[i, :] = matrix[i, :] * adj
                offset[i] = offset[i] * adj
            else:
                matrix[i, :] = 0
                offset[i] = 0
    return matrix, offset


@njit
def get_upper_relu_relax_numba(
    matrix: np.ndarray,
    offset: np.ndarray,
    size: int,
    input_lower: np.ndarray,
    input_upper: np.ndarray,
):
    # get lower and upper bounds for the bound equation
    lower = compute_min_values_numba(
        matrix=matrix, offset=offset, input_lower=input_lower, input_upper=input_upper
    )
    upper = compute_max_values_numba(
        matrix=matrix, offset=offset, input_lower=input_lower, input_upper=input_upper
    )
    # matrix and offset for the relaxation
    matrix = matrix
    offset = offset

    # compute the coefficients of the linear approximation of out
    # bound equations
    for i in range(size):
        if lower[i] >= 0:
            # Active node - Propagate lower bound equation unaltered
            pass
        elif upper[i] <= 0:
            # Inactive node - Propagate the zero function
            matrix[i, :] = 0
            offset[i] = 0
        else:
            # Unstable node - Propagate linear relaxation of
            # lower bound equations
            adj = upper[i] / (upper[i] - lower[i])
            matrix[i, :] = matrix[i, :] * adj
            offset[i] = offset[i] * adj - adj * lower[i]
    return matrix, offset


class LinearFunctions:
    """
    matrix is an (n x m) np array
    offset is an (n) np array

    An object represents n linear functions f(i) of m input variables x

    f(i) = matrix[i]*x + offset[i]

    """

    def __init__(self, matrix: np.ndarray, offset: np.ndarray):
        self.size = matrix.shape[0]
        self.matrix = matrix
        self.offset = offset

    def clone(self):
        return LinearFunctions(self.matrix.copy(), self.offset.copy())

    def get_size(self):
        return self.size

    def get_matrix(self):
        return self.matrix

    def get_offset(self):
        return self.offset

    def compute_min_max_values(self, input_lower: np.ndarray, input_upper: np.ndarray):
        weights_plus = np.maximum(self.matrix, np.zeros(self.matrix.shape))
        weights_minus = np.minimum(self.matrix, np.zeros(self.matrix.shape))
        return (
            compute_lower(weights_minus, weights_plus, input_lower, input_upper)
            + self.offset,
            compute_upper(weights_minus, weights_plus, input_lower, input_upper)
            + self.offset,
        )

    def compute_max_values(self, input_bounds):
        input_lower = input_bounds["out"]["l"]
        input_upper = input_bounds["out"]["u"]
        weights_plus = np.maximum(self.matrix, np.zeros(self.matrix.shape))
        weights_minus = np.minimum(self.matrix, np.zeros(self.matrix.shape))
        return (
            compute_upper(weights_minus, weights_plus, input_lower, input_upper)
            + self.offset
        )

    def compute_min_values(self, input_bounds):
        input_lower = input_bounds["out"]["l"]
        input_upper = input_bounds["out"]["u"]
        weights_plus = np.maximum(self.matrix, np.zeros(self.matrix.shape))
        weights_minus = np.minimum(self.matrix, np.zeros(self.matrix.shape))
        return (
            compute_lower(weights_minus, weights_plus, input_lower, input_upper)
            + self.offset
        )

    def get_lower_relu_relax(self, input_bounds):
        # get lower and upper bounds for the bound equation
        lower = self.compute_min_values(input_bounds)
        upper = self.compute_max_values(input_bounds)
        # matrix and offset for the relaxation
        matrix = self.matrix
        offset = self.offset
        # compute the coefficients of the linear approximation of out
        # bound equations
        for i in range(self.size):
            if lower[i] >= 0:
                # Active node - Propagate lower bound equation unaltered
                pass
            elif upper[i] <= 0:
                # Inactive node - Propagate the zero function
                matrix[i, :] = 0
                offset[i] = 0
            else:
                # Unstable node - Propagate linear relaxation of
                # lower bound equations

                # Standard approach
                # adj =  upper[i] / (upper[i] - lower[i])
                # matrix[i,:]  = matrix[i,:] * adj
                # offset[i]  = offset[i] * adj

                # Approach based on overapproximation areas
                # lb = lower[i]
                # ub = upper[i]
                # u_area = (pow(ub,2)/2) - (pow(ub,3)/(2*(ub-lb)))
                # l_area = (pow(lb,2) * ub) / (2*(ub-lb))
                # area = pow(ub,2)/2
                # if area < l_area + u_area:
                # matrix[i,:] = 0
                # offset[i] = 0
                # else:
                # adj =  upper[i] / (upper[i] - lower[i])
                # matrix[i,:]  = matrix[i,:] * adj
                # offset[i]  = offset[i] * adj

                # Approach based on overapproximation areas
                lb = lower[i]
                lb2 = pow(lb, 2)
                ub = upper[i]
                ub2 = pow(ub, 2)
                # over-approximation area based on double
                # linear relaxation
                area1 = ((ub * lb2) - (ub2 * lb)) / (2 * (ub - lb))
                # over-approximation area based on triangle
                # relaxation
                area2 = pow(ub, 2) / 2
                # choose the least over-approximation
                if area1 < area2:
                    adj = upper[i] / (upper[i] - lower[i])
                    matrix[i, :] = matrix[i, :] * adj
                    offset[i] = offset[i] * adj
                else:
                    matrix[i, :] = 0
                    offset[i] = 0
        return LinearFunctions(matrix, offset)

    def get_upper_relu_relax(self, input_bounds):
        # get lower and upper bounds for the bound equation
        lower = self.compute_min_values(input_bounds)
        upper = self.compute_max_values(input_bounds)
        # matrix and offset for the relaxation
        matrix = self.matrix
        offset = self.offset

        # compute the coefficients of the linear approximation of out
        # bound equations
        for i in range(self.size):
            if lower[i] >= 0:
                # Active node - Propagate lower bound equation unaltered
                pass
            elif upper[i] <= 0:
                # Inactive node - Propagate the zero function
                matrix[i, :] = 0
                offset[i] = 0
            else:
                # Unstable node - Propagate linear relaxation of
                # lower bound equations
                adj = upper[i] / (upper[i] - lower[i])
                matrix[i, :] = matrix[i, :] * adj
                offset[i] = offset[i] * adj - adj * lower[i]
        return LinearFunctions(matrix, offset)
