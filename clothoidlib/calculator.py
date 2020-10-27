"""Mini-library to calculate clothoid parameters.
"""

import numpy as np
from scipy.spatial import KDTree
from sklearn.metrics.pairwise import euclidean_distances
import typing
import functools

from .utils import ChangeOfBasis, angle_between, fresnel


class ClothoidParameters(typing.NamedTuple):
    """A named tuple for storing clothoid parameters.
    """

    gamma1: np.ndarray
    gamma2: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray
    t0: np.ndarray
    t1: np.ndarray
    t2: np.ndarray
    lambda_b: np.ndarray
    lambda_c: np.ndarray


def compute_clothoid_table(t_samples: np.ndarray) -> ClothoidParameters:
    """Calculate the clothoid parameter table for a given set of samples for t.

    Args:
        t_samples (np.ndarray): the samples

    Returns:
        ClothoidParameters: the computed clothoid parameters
    """

    # Create a matrix of all possible combinations of t1 and t2 values
    t1, t2 = np.stack(np.meshgrid(t_samples, t_samples),
                      axis=-1).reshape(-1, 2).T

    # Discard the rows where t1 < t2
    mask = (t1 < t2) & (t1 > 0)
    t1 = t1[mask]
    t2 = t2[mask]

    # Calculate points
    ts = np.stack((np.zeros_like(t1), t1, t2), axis=0)
    p0, p1, p2 = np.stack(fresnel(ts), axis=-1)

    # Calculate angles
    gamma1 = angle_between(p1-p0, p1-p2)
    gamma2 = angle_between(p2-p0, p2-p1)
    theta = np.pi * t2**2 / 2  # angle at end
    omega = np.arctan(p1[..., 1] / p1[..., 0])
    beta = omega + np.pi - gamma1 - gamma2
    alpha = theta - beta

    # Calculate t0
    lengths = np.linalg.norm(p2 - p1, axis=-1)
    t0 = np.zeros_like(t1)
    for current_t1 in t_samples:
        t1_mask = t1 == current_t1
        t2_mask = t2 == current_t1
        t1_lengths = lengths[t1_mask]
        t2_lengths = lengths[t2_mask]
        if t1_mask.sum() == 0 or t2_mask.sum() == 0:
            continue
        distance_matrix = euclidean_distances(
            t1_lengths[:, np.newaxis], t2_lengths[:, np.newaxis])
        t2_masked_indices = np.argmin(distance_matrix, axis=1)
        t2_indices = np.arange(len(t2))[t2_mask][t2_masked_indices]
        associated_t1_values = t1[t2_indices]
        t0[t1_mask] = associated_t1_values

    # Calculate lambdas
    subgoals = np.stack(fresnel(t0), axis=-1)
    lambdas = ChangeOfBasis(p1, p2)(subgoals)

    return ClothoidParameters(gamma1, gamma2, alpha, beta, t0, t1, t2, *lambdas.T)


class ClothoidCalculator:
    """Fast and efficient computation of clothoids.
    """

    def __init__(self, samples: float = 1000, t_max: float = np.sqrt(3)):
        t_samples = np.linspace(0, t_max, samples)

        # Calculate clothoid parameters
        gamma1, gamma2, *values = compute_clothoid_table(t_samples)

        # Construct kd-tree
        indices = np.array((gamma1, gamma2)).T
        self._values = np.array(values).T
        self._tree = KDTree(indices)

    def lookup_angles(self, gamma1: np.ndarray, gamma2: np.ndarray) -> ClothoidParameters:
        """Lookup clothoid parameters by providing the values of gamma1 and gamma2. 
        This method is vectorized, so when supplying arrays to gamma1 and gamma2, the parameters will be in array form too.

        Args:
            gamma1 (np.ndarray): the value of the first angle (radians)
            gamma2 (np.ndarray): the value of the second angle (radians)

        Returns:
            ClothoidParameters: the calculated parameters
        """

        # Query the kd-tree
        d, i = self._tree.query(np.stack([gamma1, gamma2], axis=-1), k=1)
        result = gamma1, gamma2, *self._values[i].T
        return ClothoidParameters(*map(np.array, result))

    def lookup_points(self, start: np.ndarray, intermediate: np.ndarray, goal: np.ndarray) -> typing.Tuple[ClothoidParameters, np.ndarray]:
        """Lookup clothoid parameters by providing a triple of points. 
        This method is vectorized, so when supplying arrays of points, the parameters will be in array form too.

        Args:
            start (np.ndarray): the starting point
            intermediate (np.ndarray): the intermediate sample point
            goal (np.ndarray): the goal point

        Returns:
            typing.Tuple[ClothoidParameters, np.ndarray]: the calculated parameters and the subgoal location
        """

        # Calculate gamma1 and gamma2
        p0, p1, p2 = goal, intermediate, start
        gamma1 = angle_between(p1-p2, p1-p0)
        gamma2 = angle_between(p2-p1, p2-p0)

        # Perform lookup
        params = self.lookup_angles(gamma1, gamma2)

        # Calculate subgoal location
        *_, lambda_b, lambda_c = params
        c = start - goal
        b = intermediate - goal
        subgoal = goal + lambda_b[..., np.newaxis] * \
            b + lambda_c[..., np.newaxis] * c

        # When gamma1 is 180° then we approximate the subgoal
        approximated_subgoal = 2 * intermediate - start
        mask = np.isclose(np.pi, gamma1)
        subgoal[mask] = approximated_subgoal[mask]

        # Return
        return params, subgoal

    def sample_clothoid(self, start: np.ndarray, intermediate: np.ndarray, goal: np.ndarray, n_samples: int = 200) -> np.ndarray:
        """Sample points along the clothoid defined by the triple <start, intermediate, goal>.

        Args:
            start (np.ndarray): the starting point
            intermediate (np.ndarray): the intermediate sample point
            goal (np.ndarray): the goal point
            n_samples (int, optional): number of samples. Defaults to 200.

        Returns:
            np.ndarray: a list of points on the clothoid
        """

        def pad(X: np.ndarray) -> np.ndarray:
            return np.hstack([X, np.ones((X.shape[0], 1))])

        def unpad(X: np.ndarray) -> np.ndarray:
            return X[:, :-1]

        def compute_transformation_matrix(I: np.ndarray, O: np.ndarray) -> np.ndarray:
            A, *_ = np.linalg.lstsq(pad(I), pad(O), rcond=None)
            return A

        def affine_transform(I: np.ndarray, A: np.ndarray):
            return unpad(pad(I) @ A)

        params, _ = self.lookup_points(start, intermediate, goal)

        gamma1, gamma2, alpha, beta, t0, t1, t2, lambda_b, lambda_c = params
        c0, c1, c2 = map(np.array, zip(
            *fresnel([np.zeros_like(t1), t1, t2])))

        p0, p1, p2 = np.array(fresnel([t0, t1, t2])).T

        P = np.array([goal, intermediate, start])
        C = np.array([c0, c1, c2])

        A = compute_transformation_matrix(C, P)
        return affine_transform(np.array(fresnel(np.linspace(0, t2, n_samples))).T, A)

    def get_clothoid_point_at_angle(self, params: ClothoidParameters, angle: float, n_samples: int = 200) -> typing.Tuple[np.ndarray, np.ndarray]:
        """Get the point on the clothoid that intersects a line drawn from the start point at a specific angle to the goal line.

        Args:
            params (ClothoidParameters): the clothoid parameters
            angle (float): the angle with the goal line
            n_samples (int, optional): the number of samples to use. Defaults to 200.

        Returns:
            typing.Tuple[np.ndarray, np.ndarray]: a tuple of (t, point)
        """

        t_samples = np.linspace(0, params.t2, n_samples)

        # remove last few samples because we don't want to consider the start point (which by definition lies on the clothoid)
        t_samples = t_samples[:-np.ceil(n_samples * .01).astype(np.int)]
        samples_x, samples_y = fresnel(t_samples)

        # the line is given by y = mx + b
        angle_with_x_axis = params.beta + angle
        print(angle_with_x_axis)
        m = np.tan(angle_with_x_axis)

        # calculate b using the fact that the line must pass through the current point: y = mx + b <=> b = y - mx
        current_point = x, y = fresnel(params.t2)
        b = y - m * x

        def distance_to_line(x, y):
            return np.abs(y - m*x - b) / np.sqrt(m**2 + 1)

        distances = distance_to_line(samples_x, samples_y)
        closest_point_index = np.argmin(distances, axis=-1)

        samples = np.array([samples_x, samples_y]).T
        return t_samples[closest_point_index], samples[closest_point_index]
