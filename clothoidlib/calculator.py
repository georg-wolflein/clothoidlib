"""Mini-library to calculate clothoid parameters.
"""

import numpy as np
from scipy.spatial import KDTree
from scipy.signal import argrelmin
from sklearn.metrics.pairwise import euclidean_distances
import typing
import functools
import logging

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

        return ClothoidParameters(gamma1, gamma2, alpha, beta, t1, t2)

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
        i_max_mask = i >= self._values.shape[0]
        if np.sum(i_max_mask) > 0:
            logger.info(
                "At least one query to the k-d tree returned an index that is out of bounds, possibly because the nearest element is the last")
            i = np.where(i_max_mask, self._values.shape[0] - 1, i)
        result = gamma1, gamma2, *self._values[i].T
        return ClothoidParameters(*map(np.array, result))

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

        # Calculate gamma1 and gamma2
        p0, p1, p2 = goal, intermediate, start
        gamma1 = angle_between(p1-p2, p1-p0)
        gamma2 = angle_between(p2-p1, p2-p0)

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
        p1 = np.array(fresnel(params.t1)).T
        p2 = np.array(fresnel(params.t2)).T
        output_space_points = np.array([intermediate, start])
        clothoid_space_points = np.array([p1, p2])
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

        clothoid_space_samples = fresnel(np.linspace(0, params.t2, n_samples))
        output_space_samples = self._project_to_output_space(start, intermediate, goal,
                                                             params, clothoid_space_samples)
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
