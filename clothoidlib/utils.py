import numpy as np
import vg
import functools
from scipy.special import fresnel as _fresnel


def fresnel(x):
    return tuple(reversed(_fresnel(x)))


angle_between = functools.partial(vg.angle, units="rad")


class ChangeOfBasis:
    """Perform a change of bases for a system of vectors.
    """

    def __init__(self, v1: np.ndarray, v2: np.ndarray):
        """Constructor.

        Args:
            v1 (np.ndarray): the first vector
            v2 (np.ndarray): the second vector
        """

        self.inverse = np.linalg.inv(np.stack((v1, v2), axis=-1))

    def __call__(self, v: np.ndarray) -> np.ndarray:
        """Transform the vector v to the new basis.

        Args:
            v (np.ndarray): the vector

        Returns:
            np.ndarray: the vector in the new basis
        """

        v = np.expand_dims(v, axis=-1)
        return np.squeeze(self.inverse @ v, axis=-1)
