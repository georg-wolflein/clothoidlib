"""Mini-library to calculate clothoid parameters.
"""

import numpy as np
from scipy.spatial import KDTree
from scipy.signal import argrelmin
from sklearn.metrics.pairwise import euclidean_distances
import typing
import functools
import logging
from pypersist import persist

from .utils import ChangeOfBasis, angle_between, fresnel

logger = logging.getLogger(__name__)


class ClothoidParameters(typing.NamedTuple):
    """A named tuple for storing clothoid parameters.
    """

    gamma1: np.ndarray
    gamma2: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray
    t1: np.ndarray
    t2: np.ndarray


class ClothoidCalculator:
    """Fast and efficient computation of clothoids.
    """

    def __init__(self, alpha_max: float = np.pi, samples: float = 1000, t_max: float = np.sqrt(5)):
        self._t_samples = np.linspace(0, t_max, samples)
        self._point_samples = fresnel(self._t_samples)

        # Calculate clothoid parameters
        gamma1, gamma2, * \
            values = params = self.compute_clothoid_table(
                self._t_samples, alpha_max)

        # Construct kd-tree
        indices = np.array((gamma1, gamma2)).T
        self._values = np.array(values).T
        self._tree = KDTree(indices)

        self._alpha_values = np.array(params).T
        self._alpha_tree = KDTree(np.expand_dims(
            np.array(params.alpha), axis=-1))

    @classmethod
    @persist
    def compute_clothoid_table(cls, t_samples: np.ndarray, alpha_max: float) -> ClothoidParameters:
        """Calculate the clothoid parameter table for a given set of samples for t.

        Args:
            t_samples (np.ndarray): the samples
            alpha_max (float): maximum value of alpha

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

        # Discard rows where alpha >= 2 * np.pi
        mask = alpha <= alpha_max
        params = ClothoidParameters(gamma1, gamma2, alpha, beta, t1, t2)
        params = ClothoidParameters(*(x[mask] for x in params))
        return params

    def lookup_alpha(self, alpha: np.ndarray) -> ClothoidParameters:
        """Lookup clothoid parameters by providing alpha.

        Args:
            alpha (np.ndarray): the value(s) of alpha

        Returns:
            ClothoidParameters: the clothoid params
        """

        squeezed = len(alpha.shape) == 0
        if squeezed:
            alpha = np.expand_dims(alpha, axis=0)

        # Query the kd-tree
        straight_line_mask = alpha == 0
        d, i = self._alpha_tree.query(np.expand_dims(alpha, axis=-1), k=1)
        i_max_mask = i >= self._alpha_values.shape[0]
        if np.sum(i_max_mask) > 0:
            logger.info(
                "At least one query to the k-d tree returned an index that is out of bounds, possibly because the nearest element is the last")
            i = np.where(i_max_mask, self._alpha_values.shape[0] - 1, i)
        values = np.zeros((alpha.shape[0], 6))
        values[~straight_line_mask] = self._alpha_values[i]
        result = values.T
        if squeezed:
            result = map(functools.partial(np.squeeze, axis=0), result)
        return ClothoidParameters(*result)

    def lookup_angles(self, gamma1: np.ndarray, gamma2: np.ndarray) -> ClothoidParameters:
        """Lookup clothoid parameters by providing the values of gamma1 and gamma2.
        This method is vectorized, so when supplying arrays to gamma1 and gamma2, the parameters will be in array form too.

        Args:
            gamma1 (np.ndarray): the value of the first angle (radians)
            gamma2 (np.ndarray): the value of the second angle (radians)

        Returns:
            ClothoidParameters: the calculated parameters
        """

        if gamma1.shape != gamma2.shape:
            raise ValueError("shape mismatch")

        squeezed = len(gamma1.shape) == 0
        if squeezed:
            gamma1 = np.expand_dims(gamma1, axis=0)
            gamma2 = np.expand_dims(gamma2, axis=0)
        straight_line_mask = (gamma1 == 0) & (gamma2 == 0)
        gammas = np.stack([gamma1, gamma2], axis=-1)[~straight_line_mask]

        # Query the kd-tree
        d, i = self._tree.query(gammas, k=1)
        i_max_mask = i >= self._values.shape[0]
        if np.sum(i_max_mask) > 0:
            logger.info(
                "At least one query to the k-d tree returned an index that is out of bounds, possibly because the nearest element is the last")
            i = np.where(i_max_mask, self._values.shape[0] - 1, i)
        values = np.zeros((gamma1.shape[0], 4))
        values[~straight_line_mask] = self._values[i]
        result = gamma1, gamma2, *values.T
        result = map(np.array, result)
        if squeezed:
            result = map(functools.partial(np.squeeze, axis=0), result)

        return ClothoidParameters(*result)

    def lookup_points(self, start: np.ndarray, intermediate: np.ndarray, goal: np.ndarray) -> ClothoidParameters:
        """Lookup clothoid parameters by providing a triple of points.
        This method is vectorized, so when supplying arrays of points, the parameters will be in array form too.

        Args:
            start (np.ndarray): the starting point
            intermediate (np.ndarray): the intermediate sample point
            goal (np.ndarray): the goal point

        Returns:
            ClothoidParameters: the calculated parameters
        """

        start, intermediate, goal = np.broadcast_arrays(
            start, intermediate, goal)
        straight_line_mask = np.all(start == intermediate, axis=-1)
        batch_dimensions = max(map(np.shape, (start, intermediate, goal)),
                               key=len)[:-1]

        # Calculate gamma1 and gamma2
        p0, p1, p2 = goal, intermediate, start
        p0 = p0[~straight_line_mask]
        p1 = p1[~straight_line_mask]
        p2 = p2[~straight_line_mask]
        gamma1 = np.zeros(batch_dimensions)
        gamma2 = np.zeros(batch_dimensions)
        gamma1[~straight_line_mask] = angle_between(p1-p2, p1-p0)
        gamma2[~straight_line_mask] = angle_between(p2-p1, p2-p0)

        # Perform lookup
        params = self.lookup_angles(gamma1, gamma2)

        # Return
        return params

    def _project_to_output_space(self, start: np.ndarray, intermediate: np.ndarray, goal: np.ndarray, params: ClothoidParameters, points_in_clothoid_space: np.ndarray) -> np.ndarray:
        """Transform points in clothoid space to output space.

        Args:
            start (np.ndarray): the starting point
            intermediate (np.ndarray): the intermediate sample point
            goal (np.ndarray): the goal point
            params (ClothoidParameters): the clothoid parameters
            points_in_clothoid_space (np.ndarray): points in clothoid space that are to be transformed

        Returns:
            np.ndarray: the transformed points in output space
        """

        # Translate output space so that the goal is at the origin
        start = start - goal
        intermediate = intermediate - goal

        # We need to find the transformation matrix from clothoid space to output space
        p1 = fresnel(params.t1)
        p2 = fresnel(params.t2)
        output_space_points = np.stack([intermediate, start], axis=-2)
        clothoid_space_points = np.stack([p1, p2], axis=-2)
        M = np.linalg.solve(clothoid_space_points, output_space_points)

        points_in_output_space = goal + points_in_clothoid_space @ M
        return points_in_output_space

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

        params = self.lookup_points(start, intermediate, goal)

        zeros_mask = params.t2 == 0
        batch_dimensions = params.t2.shape
        n_dimensions = start.shape[-1]
        output_space_samples = np.zeros((*batch_dimensions,
                                         n_samples, n_dimensions))

        clothoid_space_samples = fresnel(np.linspace(0, params.t2[~zeros_mask],
                                                     n_samples))
        clothoid_space_samples = np.moveaxis(clothoid_space_samples, 0, -2)
        if np.sum(~zeros_mask) > 0:
            output_space_samples[~zeros_mask] = self._project_to_output_space(start[~zeros_mask],
                                                                              intermediate[~zeros_mask],
                                                                              goal[~zeros_mask],
                                                                              ClothoidParameters(
                *(x[~zeros_mask] for x in params)),
                clothoid_space_samples)

        if np.sum(zeros_mask) > 0:
            output_space_samples[zeros_mask] = start[zeros_mask] + \
                (goal[zeros_mask] - start[zeros_mask]) * \
                np.linspace(0., 1., n_samples)[..., np.newaxis]
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
        local_minima, = argrelmin(distances, axis=-1)
        closest_point_index = local_minima[0] if len(
            local_minima) > 0 else np.argmin(distances, axis=-1)

        return t_samples[closest_point_index], point_samples[closest_point_index]
