import numpy as np

from clothoidlib.calculator import ClothoidCalculator

calculator = ClothoidCalculator()


def test_get_clothoid_point_at_angle():
    start = np.array([0., 0])
    intermediate = np.array([1., 1.])
    goal = np.array([5., 0])

    params, subgoal = calculator.lookup_points(start, intermediate, goal)
    gamma1, gamma2, alpha, beta, t0, t1, t2, lambda_b, lambda_c = params
    angle = np.pi / 6  # 30°
    t, point = calculator.get_clothoid_point_at_angle(params, angle)

    assert 0 < t < t2
