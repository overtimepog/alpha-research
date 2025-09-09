import numpy as np
import scipy as sp


def cal_ratio(construction_1):
    pairwise_distances = sp.spatial.distance.pdist(construction_1)
    min_distance = np.min(pairwise_distances)
    max_distance = np.max(pairwise_distances)
    ratio_squared = (min_distance / max_distance)**2
    return ratio_squared


def evaluate(program_path: str = "results/initial_program.py"):
    """
    Evaluate the pack_circles function from the given program file.
    Returns the total radius sum if valid, otherwise raises an exception.
    """
    import importlib.util
    import sys
    
    # Load the module from the given path
    spec = importlib.util.spec_from_file_location("program", program_path)
    program = importlib.util.module_from_spec(spec)
    sys.modules["program"] = program
    spec.loader.exec_module(program)


    # Check if 'max_min_dis_ratio' exists in the loaded module
    if not hasattr(program, 'max_min_dis_ratio'):
        raise ValueError(f"The file '{program_path}' does not define 'max_min_dis_ratio'.")

    try:
        res_n16_d2, _ = program.max_min_dis_ratio(16, 2) 
        res_n14_d3, _ = program.max_min_dis_ratio(14, 3)
    except Exception as e1:
        return {"result": -10.0, "error": e1}
    
    try:
        ratio_n16_d2 = cal_ratio(res_n16_d2) # AlphaEvolve: 1 / 12.88926611203463 = 0.07758393622320406
        ratio_n14_d3 = cal_ratio(res_n14_d3) # AlphaEvolve: 1 / 4.165849767 = 0.24004706263470807


    except Exception as e:
        return {"result": -1.0, "error": e}

    results = {
        "ratio_n16_d2": ratio_n16_d2,
        "ratio_n14_d3": ratio_n14_d3, 
    }
    
    return results

print(evaluate())