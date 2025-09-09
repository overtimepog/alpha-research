import numpy as np
from scipy.spatial.distance import pdist
from scipy.spatial import cKDTree

# (Removed) smooth_points — smoothing logic is now inlined to reduce indirection


def calculate_distances(points):
    """Calculates min, max, and ratio of pairwise Euclidean distances using scipy pdist."""
    if points.shape[0] < 2:
        return 0.0, 0.0, 0.0
    distances = pdist(points, metric='euclidean')
    eps = 1e-8
    min_dist = max(np.min(distances), eps)
    max_dist = np.max(distances)
    ratio = max_dist / min_dist
    return min_dist, max_dist, ratio

# (Removed) perturb_point — now inlined directly where used

def update_temperature(temperature, cooling_rate, accept_history, iteration, total_iters, initial_temperature, window_size=100):
    """
    Adaptive cooling with acceptance‐rate feedback and periodic reheating.
    """
    window = accept_history[-min(len(accept_history), window_size):]
    rate = sum(window) / len(window)
    # gentler correction: slow/fast cooling factors reduced
    if rate < 0.2:
        adj = 1.02
    elif rate > 0.8:
        adj = 0.98
    else:
        adj = 1.0
    temperature *= cooling_rate * adj
    # removed periodic reheating to maintain smoother cooling schedule
    # if (iteration + 1) % (total_iters // 4) == 0:
    #     temperature = initial_temperature
    return temperature

def max_min_dis_ratio(n: int, d: int, seed=None):
    """
    Finds n points in d-dimensional space to minimize the max/min distance ratio 
    using simulated annealing.

    Args:
        n (int): Number of points.
        d (int): Dimensionality of the space.

    Returns:
        tuple: (best_points, best_ratio)
    """

    # Adaptive hyperparameters based on dimensionality
    iterations = 3000 if d <= 2 else 6000  # increased sweeps for improved convergence
    initial_temperature = 10.0
    cooling_rate = 0.998 if d <= 2 else 0.996  # slower cooling for extended exploration
    perturbation_factor = 0.15 if d <= 2 else 0.12  # tuned smaller steps in 3D for better local refinement
    # relaxation factor for post-acceptance repulsive adjustment
    # relaxation_factor removed; using inline 0.1 * perturbation_factor below

    # 1. Initial State: reproducible random generator
    rng = np.random.default_rng(seed)
    # uniform random initialization in [0,1]^d for simplicity
    current_points = rng.random((n, d))
    
    _, _, current_ratio = calculate_distances(current_points)
    
    best_points = np.copy(current_points)
    best_ratio = current_ratio
    
    temperature = initial_temperature
    accept_history = []
    window_size = 50  # window for stagnation detection and adaptive injection
    # smoothing_interval remains, but smoothing_strength is fixed inlined above
    smoothing_interval = max(10, iterations // (20 if d <= 2 else 30))  # more frequent smoothing in 3D for improved uniformity
    
    for i in range(iterations):
        # Build KD-tree once per iteration for neighbor queries
        tree = cKDTree(current_points)
        # optional smoothing step using distance-weighted neighbor smoothing
        if (i + 1) % smoothing_interval == 0:
            # choose neighbor count based on dimension
            k_smooth = 6 if d > 2 else 4
            _, idxs = tree.query(current_points, k=k_smooth+1)
            neighbors = current_points[idxs[:,1:]]  # exclude self
            # compute inverse-distance weights
            diffs = neighbors - current_points[:, None, :]
            dists = np.linalg.norm(diffs, axis=2) + 1e-6
            weights = 1.0 / dists
            weights /= weights.sum(axis=1, keepdims=True)
            neighbor_means = (neighbors * weights[..., None]).sum(axis=1)
            blend = 0.6 if d > 2 else 0.7
            current_points = np.clip(current_points * blend + neighbor_means * (1 - blend), 0.0, 1.0)
            _, _, current_ratio = calculate_distances(current_points)
            if current_ratio < best_ratio:
                best_points = current_points.copy()
                best_ratio = current_ratio

        # 2. Generate Neighboring State: Perturb a random point
        # Simplify scaling: rely on temperature to adjust step-size instead of best_ratio
        # dynamic perturbation decays sublinearly with temperature for finer local moves
        perturbation_strength = perturbation_factor * ((temperature / initial_temperature)**0.6 + 0.15)
        
        # Choose a random point to perturb
        point_to_perturb_index = rng.integers(0, n)
        
        old_point = current_points[point_to_perturb_index].copy()
        # Increase repulsive‐move frequency in low dimensions
        # dynamic repulsion probability: stronger at high temperature, tapering off as we cool
        if d > 2:
            # reduce repulsion frequency in 3D for finer refinement
            repulsion_prob = float(np.clip(temperature / initial_temperature, 0.2, 0.8))
        else:
            repulsion_prob = float(np.clip(temperature / initial_temperature + 0.1, 0.5, 0.95))
        # start with a random jitter
        # random jitter inlined for readability
        candidate = old_point + rng.uniform(-perturbation_strength, perturbation_strength, size=old_point.shape)
        if n > 1 and rng.random() < repulsion_prob:
            # compute nearest neighbor via KD-tree for efficiency (reusing prebuilt tree)
            _, nn_idxs = tree.query(old_point, k=2)
            nn_idx = nn_idxs[1]
            vec = old_point - current_points[nn_idx]
            norm = np.linalg.norm(vec)
            if norm > 1e-8:
                dir_vec = vec / norm
                candidate = old_point + perturbation_strength * dir_vec
        # keep the point in [0,1]^d
        current_points[point_to_perturb_index] = np.clip(candidate, 0.0, 1.0)
        _, _, candidate_ratio = calculate_distances(current_points)
        
        # Acceptance criterion
        delta = candidate_ratio - current_ratio
        accept = (delta < 0) or (rng.random() < np.exp(-delta / temperature))

        if accept:
            current_ratio = candidate_ratio
            # Post-acceptance repulsive relaxation to improve local spacing
            # reuse prebuilt KD-tree for repulsive relaxation
            dists, idxs_nn = tree.query(current_points[point_to_perturb_index], k=2)
            dir_vec = current_points[point_to_perturb_index] - current_points[idxs_nn[1]]
            norm = np.linalg.norm(dir_vec)
            if norm > 1e-8:
                # push away from nearest neighbor
                adjustment = 0.1 * perturbation_factor * dir_vec / norm
                current_points[point_to_perturb_index] = np.clip(
                    current_points[point_to_perturb_index] + adjustment, 0.0, 1.0
                )
                # update ratio and best points after relaxation
                _, _, relaxed_ratio = calculate_distances(current_points)
                current_ratio = relaxed_ratio
                if relaxed_ratio < best_ratio:
                    best_points = current_points.copy()
                    best_ratio = relaxed_ratio
            # also keep the standard best‐check for the candidate move
            if current_ratio < best_ratio:
                best_points = current_points.copy()
                best_ratio = current_ratio
        else:
            current_points[point_to_perturb_index] = old_point
        
        # Update temperature with adaptive schedule
        accept_history.append(accept)
        temperature = update_temperature(temperature, cooling_rate, accept_history, i, iterations, initial_temperature)
        # periodic mild reheating for 3D to escape deep minima
        if d > 2 and (i + 1) % (iterations // 3) == 0:
            temperature = max(temperature, initial_temperature * 0.3)

        # random injection to escape plateaus: reinitialize one point every 20% of iterations
        # random injection only if we’ve stagnated (low acceptance in recent window)
        if (i + 1) % max(1, iterations // 5) == 0 and len(accept_history) >= window_size \
           and sum(accept_history[-window_size:]) / window_size < 0.1:
            j = rng.integers(0, n)
            current_points[j] = rng.random(d)
            _, _, current_ratio = calculate_distances(current_points)

    # Local refinement stage: fine-tune best solution with small Gaussian perturbations
    refine_iters = max(100, iterations // 20)
    for _ in range(refine_iters):
        idx = rng.integers(0, n)
        old_point = best_points[idx].copy()
        perturb = rng.normal(0, perturbation_factor * 0.05, size=d)
        best_points[idx] = np.clip(old_point + perturb, 0.0, 1.0)
        _, _, refined_ratio = calculate_distances(best_points)
        if refined_ratio < best_ratio:
            best_ratio = refined_ratio
        else:
            best_points[idx] = old_point
    return best_points, best_ratio