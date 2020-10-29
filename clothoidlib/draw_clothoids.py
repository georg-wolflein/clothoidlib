"""Proof of concept script to draw clothoids.
"""

import numpy as np
import matplotlib.pyplot as plt
import typing
from clothoidlib import ClothoidCalculator
import argparse

if __name__ == "__main__":
    # Parse command line args
    parser = argparse.ArgumentParser(description="Clothoid drawing utility")
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
