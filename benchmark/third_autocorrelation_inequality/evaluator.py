import numpy as np

def evaluate(program_path: str = "/data/zhuotaodeng/yzj/_para/alpha-research/math/initial_program.py"):
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
    try:
        height_sequence_3 = program.find_better_c3_upper_bound()
    except:
        return {"error": -10.0}
    
    convolution_3 = np.convolve(height_sequence_3, height_sequence_3)
    C_upper_bound = abs(2 * len(height_sequence_3) * np.max(convolution_3) / (np.sum(height_sequence_3)**2))
    
    return {"1 / C_upper_bound": 1 / C_upper_bound} # 0.6869

print(evaluate())