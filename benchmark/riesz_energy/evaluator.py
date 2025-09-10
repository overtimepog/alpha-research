import sys
import os
import json
import importlib.util
import numpy as np
from typing import Dict

EPS = 1e-12

def evaluate_riesz_energy(points: np.ndarray, s: float = 1.0) -> Dict[str, float]:
    xs = np.clip(np.asarray(points, dtype=float).ravel(), 0.0, 1.0)
    xs.sort()
    n = len(xs)
    if n < 2:
        return {"valid": 0.0, "energy": float("inf"), "min_spacing": 0.0}
    energy = 0.0
    dmin = float("inf")
    for i in range(n):
        xi = xs[i]
        for j in range(i+1, n):
            d = abs(xi - xs[j])
            dmin = min(dmin, d)
            energy += 1.0 / (d**s + EPS)
    return {"valid": 1.0, "energy": float(energy), "min_spacing": float(dmin)}

def evaluate(program_path: str):
    try:
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        sys.modules["program"] = program
        spec.loader.exec_module(program)

        pts = None
        if hasattr(program, 'xs'):
            pts = program.xs
        elif hasattr(program, 'main'):
            res = program.main()
            if isinstance(res, np.ndarray):
                pts = res
            elif hasattr(program, 'xs'):
                pts = program.xs
        if pts is None:
            return -1.0
        result = evaluate_riesz_energy(pts, s=1.0)
        if result.get("valid", 0.0) != 1.0:
            return -1.0
        return float(result["energy"])
    except Exception:
        return -1.0

if __name__ == "__main__":
    try:
        default_path = os.path.join(os.path.dirname(__file__), "initial_program.py")
    except Exception:
        default_path = "initial_program.py"
    target = sys.argv[1] if len(sys.argv) > 1 else default_path
    print(json.dumps(evaluate(target)))
