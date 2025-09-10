import numpy as np
import importlib.util
import sys
import os
import json
from itertools import combinations
from typing import Dict

def _triangle_area(a, b, c) -> float:
    return abs((b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])) * 0.5

def evaluate_min_triangle_area(points: np.ndarray) -> Dict[str, float]:
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    n = len(pts)
    if n < 3:
        return {"valid": 1.0, "min_area": 0.0, "n": float(n)}
    min_area = float("inf")
    for i, j, k in combinations(range(n), 3):
        area = _triangle_area(pts[i], pts[j], pts[k])
        if area < min_area:
            min_area = area
    return {"valid": 1.0, "min_area": float(min_area), "n": float(n)}

def evaluate(program_path: str):
    try:
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        sys.modules["program"] = program
        spec.loader.exec_module(program)

        points = None
        if hasattr(program, 'points'):
            points = program.points
        elif hasattr(program, 'main'):
            res = program.main()
            if isinstance(res, np.ndarray):
                points = res
            elif hasattr(program, 'points'):
                points = program.points
        if points is None:
            return {"error": -1.0}
        result = evaluate_min_triangle_area(points)
        return {"min_area": result["min_area"], "n": result["n"]}
    except Exception:
        return {"error": -1.0}

if __name__ == "__main__":
    try:
        default_path = os.path.join(os.path.dirname(__file__), "initial_program.py")
    except Exception:
        default_path = "initial_program.py"
    target = sys.argv[1] if len(sys.argv) > 1 else default_path
    print(json.dumps(evaluate(target), ensure_ascii=False, indent=2))
