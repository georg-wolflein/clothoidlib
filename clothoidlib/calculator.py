"""Mini-library to calculate clothoid parameters.
"""

import numpy as np
from scipy.spatial import KDTree
from scipy.signal import argrelmin
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


class ClothoidCalculator:
    """Fast and efficient computation of clothoids.
    """

    def __init__(self, samples: float = 1000, t_max: float = np.sqrt(3)):
        self._t_samples = np.linspace(0, t_max, samples)
        self._point_samples = fresnel(self._t_samples)

        # Calculate clothoid parameters
        gamma1, gamma2, *values = self.compute_clothoid_table(self._t_samples)

        # Construct kd-tree
        indices = np.array((gamma1, gamma2)).T
        self._values = np.array(values).T
        self._tree = KDTree(indices)

    @classmethod
    def compute_clothoid_table(cls, t_samples: np.ndarray) -> ClothoidParameters:
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
        p0, p1, p2 = fresnel(ts)

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
        subgoals = fresnel(t0)
        lambdas = ChangeOfBasis(p1, p2)(subgoals)

        return ClothoidParameters(gamma1, gamma2, alpha, beta, t0, t1, t2, *lambdas.T)

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

        params, _ = self.lookup_points(start, intermediate, goal)

        gamma1, gamma2, alpha, beta, t0, t1, t2, lambda_b, lambda_c = params
        c1 = np.array(fresnel(t1)).T
        c2 = np.array(fresnel(t2)).T

        # Translate output space so that the goal is at the origin
        start -= goal
        intermediate -= goal

        output_space_points = np.array([intermediate, start])
        clothoid_space_points = np.array([c1, c2])

        # We need to find the transformation matrix from clothoid space to output space
        M = np.linalg.solve(clothoid_space_points, output_space_points)

        clothoid_space_samples = np.array(
            fresnel(np.linspace(0, t2, n_samples))).T
        output_space_samples = goal + clothoid_space_points @ M
        return output_space_samples

    def get_clothoid_point_at_angle(self, params: ClothoidParameters, angle: float) -> typing.Tuple[float, float]:
        """Get the point on the clothoid that intersects a line drawn from the start point at a specific angle to the goal line.

        Args:
            params (ClothoidParameters): the clothoid parameters
            angle (np.ndarray): the angle with the goal line

        Returns:
            typing.Tuple[np.ndarray, np.ndarray]: a tuple of (t, point)
        """

        n_samples = np.argmax(self._t_samples > params.t2)
        t_samples = self._t_samples[:n_samples]
        point_samples = self._point_samples[:n_samples]

        samples_x, samples_y = point_samples.T

        # the line is given by y = mx + b
        angle_with_x_axis = params.beta + angle
        m = np.tan(angle_with_x_axis)

        # calculate b using the fact that the line must pass through the current point: y = mx + b <=> b = y - mx
        current_point = x, y = fresnel(params.t2)
        b = y - m * x

        # If angle > alpha, we are certain that the closest point is the current point
        if not (0 < angle < params.alpha):
            return params.t2, current_point

        # Calculate distance of each point on clothoid to the line
        def distance_to_line(x, y):
            return np.abs(y - m*x - b) / np.sqrt(m**2 + 1)

        distances = distance_to_line(samples_x, samples_y)

        # Find local minima in distances
        local_minima = argrelmin(distances, axis=-1)
        closest_point_index = local_minima[0] if len(
            local_minima) > 0 else np.argmin(distances, axis=-1)

        return t_samples[closest_point_index], point_samples[closest_point_index]

    def calculate_point_distance_to_clothoid(self, start: np.ndarray, intermediate: np.ndarray, goal: np.ndarray, point: np.ndarray) -> np.ndarray:
        pass
