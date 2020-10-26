import numpy as np
from scipy.special import fresnel as _fresnel


def fresnel(x):
    return tuple(reversed(_fresnel(x)))


def angle_between(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Calculate the angle between two n-dimensional vectors.
    This method is vectorized, so when supplying matrices, they will be interpreted as collections of vectors.

    Args:
        v1 (np.ndarray): the first vector
        v2 (np.ndarray): the second vector

    Returns:
        np.ndarray: the calculated angle(s)
    """

    if v1.size == 0 or v2.size == 0:
        return np.array([], dtype=np.float)
    v1 /= np.linalg.norm(v1, axis=-1, keepdims=True)
    v2 /= np.linalg.norm(v2, axis=-1, keepdims=True)
    v1 = np.expand_dims(v1, axis=-2)
    v2 = np.expand_dims(v2, axis=-1)
    return np.arccos(np.clip(np.matmul(v1, v2), -1., 1.)).squeeze(axis=(-1, -2))


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
