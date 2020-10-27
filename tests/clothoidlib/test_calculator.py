import numpy as np

from clothoidlib.calculator import ClothoidCalculator
from clothoidlib.utils import fresnel, angle_between

calculator = ClothoidCalculator()


def test_get_clothoid_point_at_angle():
    start = np.array([0., 0])
    intermediate = np.array([1., 1.])
    goal = np.array([5., 0])

    params = calculator.lookup_points(start, intermediate, goal)
    angle = np.pi / 6  # 30°
    t, point = calculator.get_clothoid_point_at_angle(params, angle)

    assert 0 < t < params.t2


def test_sample_clothoid():
    t1 = .5
    t2 = .8

    start = fresnel(t2)
    intermediate = fresnel(t1)
    goal = np.array([0., 0])
    n_samples = 200

    points = fresnel(np.linspace(0, t2, n_samples))
    result = calculator.sample_clothoid(
        start, intermediate, goal, n_samples=n_samples)

    mse = np.mean((result - points)**2)
    assert mse < 1e-5
