"""Proof of concept script to draw clothoids.
"""

import numpy as np
import matplotlib.pyplot as plt
import typing
from clothoidlib import ClothoidCalculator
from clothoidlib.utils import angle_between
import argparse

if __name__ == "__main__":
    # Parse command line args
    parser = argparse.ArgumentParser(description="Clothoid drawing utility")
    parser.add_argument("--intersection", action="store_true",
                        help="find intersection point")
    parser.set_defaults(intersection=False)
    args = parser.parse_args()

    # Create calculator
    calculator = ClothoidCalculator()

    # Set up plot
    fig = plt.figure()
    plt.axis("equal")
    plt.plot([0, 1], [0, 1], c="w")
    plt.show(block=False)

    # Continually wait for user input
    while True:

        # Get user to add three points
        print("Click on three points on the plot, and a clothoid will be drawn.")
        points = np.array(plt.ginput(3, timeout=0))
        if points.shape != (3, 2):
            continue

        # Draw points
        plt.plot(*calculator.sample_clothoid(*points).T, c="r")
        plt.plot(*np.array([*points, points[0]]).T, c="k")
        plt.show(block=False)
        fig.canvas.draw_idle()

        if args.intersection:
            print("Click on a point on the plot, and I will compute the intersection of the straight line from the start to that point")
            point = np.array(plt.ginput(1, timeout=0))
            if point.shape != (1, 2):
                continue
            params = calculator.lookup_points(*points)
            start, intermediate, goal = points
            angle = angle_between(goal-start, point-start)
            t, intersection_point = calculator.get_clothoid_point_at_angle(
                params, angle)
            plt.scatter(*point.T, c="g")
            plt.scatter(*calculator._project_to_output_space(start,
                                                             intermediate, goal, params, intersection_point).T, c="b")
            plt.show(block=False)
            fig.canvas.draw_idle()
