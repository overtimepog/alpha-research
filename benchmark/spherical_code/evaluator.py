import sys
import os
import json
import numpy as np
from typing import Dict
import importlib.util

EPS = 1e-12

def evaluate_spherical_code_min_angle(points: np.ndarray) -> Dict[str, float]:
    P = np.asarray(points, dtype=float)
    if P.ndim != 2 or P.shape[0] < 2:
        return {"valid": 0.0, "min_angle": 0.0, "n": 0.0, "dimension": 0.0, "score": 0.0}
    # normalize rows onto the sphere
    norms = np.maximum(np.linalg.norm(P, axis=1, keepdims=True), EPS)
    P = P / norms
    n = P.shape[0]
    d = P.shape[1]
    min_angle = float("inf")
    for i in range(n):
        for j in range(i+1, n):
            cosang = float(np.clip(np.dot(P[i], P[j]), -1.0, 1.0))
            ang = float(np.arccos(cosang))
            if ang < min_angle:
                min_angle = ang
    return {"valid": 1.0, "min_angle": float(min_angle), "n": float(n), "dimension": float(d), "score": float(min_angle)}

def evaluate(program_path: str):
    try:
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        sys.modules["program"] = program
        spec.loader.exec_module(program)

        pts = None
        if hasattr(program, 'points'):
            pts = program.points
        elif hasattr(program, 'main'):
            res = program.main()
            if isinstance(res, np.ndarray):
                pts = res
            elif hasattr(program, 'points'):
                pts = program.points
        if pts is None:
            return {"error": -1.0}
        result = evaluate_spherical_code_min_angle(pts)
        return {"score": result["score"], "min_angle": result["min_angle"], "n": result["n"], "dimension": result["dimension"]}
    except Exception:
        return {"error": -1.0}

if __name__ == "__main__":
    try:
        default_path = os.path.join(os.path.dirname(__file__), "initial_program.py")
    except Exception:
        default_path = "initial_program.py"
    target = sys.argv[1] if len(sys.argv) > 1 else default_path
    print(json.dumps(evaluate(target), ensure_ascii=False, indent=2))
