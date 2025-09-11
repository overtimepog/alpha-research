import math
import numpy as np
import itertools
import random

random.seed(42)
np.random.seed(42)


def verify_circles(circles):
    """Checks that the circles are disjoint and lie inside a unit square.

    Args:
      circles: A list of tuples (x, y, radius) or numpy array of shape (num_circles, 3)

    Returns:
      bool: True if circles are valid (disjoint and within unit square), False otherwise
    """
    # Convert to numpy array if it's a list
    if not isinstance(circles, np.ndarray):
        circles = np.array(circles)
    
    # Check pairwise disjointness
    for circle1, circle2 in itertools.combinations(circles, 2):
        center_distance = np.sqrt((circle1[0] - circle2[0])**2 + (circle1[1] - circle2[1])**2)
        radii_sum = circle1[2] + circle2[2]
        if center_distance < radii_sum:
            return False

    # Check all circles lie inside the unit square [0,1]x[0,1]
    for circle in circles:
        if not (0 <= min(circle[0], circle[1]) - circle[2] and max(circle[0], circle[1]) + circle[2] <= 1):
            return False
    
    return True


def evaluate(program_path: str = "results/initial_program.py"):
    """
    Evaluate the pack_circles function from the given program file.
    Returns dict with keys: score, result_26, result_32; score is sum of totals.
    """
    import importlib.util
    import sys
    
    # Load the module from the given path
    spec = importlib.util.spec_from_file_location("program", program_path)
    program = importlib.util.module_from_spec(spec)
    sys.modules["program"] = program
    spec.loader.exec_module(program)
    
    # Test the pack_circles function
    try:
        total_r_26, circles_26 = program.pack_circles(26)
        total_r_32, circles_32 = program.pack_circles(32)
    except Exception as e:
        return {"error": -10.0}
    
    # Validate the circles
    valid_26 = verify_circles(circles_26)
    valid_32 = verify_circles(circles_32)

    if not all((valid_26, valid_32)):
        return {"error": -1.0}
    
    score = float(total_r_26 + total_r_32)
    return {
        "score": score,
        "result_26": total_r_26,
        "result_32": total_r_32
    }

print(evaluate("/data/zhuotaodeng/yzj/_para/alpha-research/results_circles_1/test.py"))