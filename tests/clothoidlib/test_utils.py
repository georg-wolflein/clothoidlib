import numpy as np

from clothoidlib.utils import angle_between


def test_angle_between_2d_vectors():
    v1 = np.array([1., 0])
    v2 = np.array([0., 1])
    result = angle_between(v1, v2)
    assert result.shape == tuple()
    assert result == np.pi / 2


def test_angle_between_multiple_2d_vectors():
    v1 = np.array([[1., 0], [0, 1]])
    v2 = np.array([[0., 1], [1, 1]])
    expected = np.array([np.pi / 2, np.pi / 4])
    result = angle_between(v1, v2)
    assert result.shape == (2,)
    assert np.allclose(result, expected)


def test_angle_between_3d_vectors():
    v1 = np.array([1., 0, 0])
    v2 = np.array([0., 1, 0])
    assert angle_between(v1, v2) == np.pi / 2
