import math
import random
from concurrent.futures import ThreadPoolExecutor


def pack_circles(n, square_size=1.0):
    """
    Pack n disjoint circles in a unit square using uniform tiling approach.
    Returns the sum of radii and list of circles (x, y, r).
    """

    def max_circle_radius(x, y, circles, square_size=1.0, skip_idx=None):
        """
        Compute the maximum radius for a circle centered at (x, y) that:
        - Stays within the unit square [0, square_size] × [0, square_size].
        - Does not overlap with existing circles.
        skip_idx: if provided, index in circles[] to ignore (self).
        """
        # Distance to nearest boundary of the unit square
        r_max = min(x, y, square_size - x, square_size - y)
        
        # Check distance to existing circles, exit early if r_max → 0
        # early exit if r_max is tiny, and avoid needless sqrt
        for idx, (cx, cy, cr) in enumerate(circles):
            if skip_idx == idx:
                continue
            if r_max <= 1e-8:
                break
            dx = x - cx
            dy = y - cy
            sep = r_max + cr
            if dx*dx + dy*dy < sep*sep:
                # only compute sqrt when we know we can shrink
                dist = math.sqrt(dx*dx + dy*dy)
                r_max = min(r_max, dist - cr)
        return max(r_max, 0.0)

    def uniform_tiling_circles(n, square_size=1.0):
        """
        Uniformly tile the square with circles using optimal grid placement.
        """
        if n <= 0:
            return []
        
        circles = []
        
        # Calculate optimal grid dimensions
        # For n circles, find the best grid layout (rows x cols)
        best_layout = None
        best_total_radius = 0
        
        # Try different grid configurations
        for rows in range(1, min(n + 1, 20)):
            cols = math.ceil(n / rows)
            if cols > 20:  # Limit grid size
                continue
                
            # Calculate spacing
            spacing_x = square_size / (cols + 1)
            spacing_y = square_size / (rows + 1)
            
            # Use the smaller spacing to ensure circles fit
            min_spacing = min(spacing_x, spacing_y)
            
            # Calculate maximum radius for this layout
            max_radius = min_spacing / 2
            
            # Ensure radius doesn't exceed boundaries
            max_radius = min(max_radius, 
                           spacing_x / 2 - 1e-6, 
                           spacing_y / 2 - 1e-6)
            
            if max_radius <= 0:
                continue
            
            # Place circles in uniform grid
            temp_circles = []
            count = 0
            
            for row in range(rows):
                for col in range(cols):
                    if count >= n:
                        break
                    
                    x = spacing_x * (col + 1)
                    y = spacing_y * (row + 1)
                    
                    # Ensure circle stays within bounds
                    if (x - max_radius >= 0 and x + max_radius <= square_size and
                        y - max_radius >= 0 and y + max_radius <= square_size):
                        
                        temp_circles.append((x, y, max_radius))
                        count += 1
                
                if count >= n:
                    break
            
            # Calculate total radius for this layout
            total_radius = len(temp_circles) * max_radius
            
            if total_radius > best_total_radius and len(temp_circles) == n:
                best_total_radius = total_radius
                best_layout = temp_circles
        
        # If we found a valid layout, return it
        if best_layout:
            return best_layout
        
        # Fallback: use hexagonal packing for better density
        return hexagonal_packing(n, square_size)

    def hexagonal_packing(n, square_size=1.0):
        """
        Use hexagonal close packing for better space utilization.
        """
        circles = []
        
        # Estimate number of rows and columns for hexagonal packing
        # Hexagonal packing has rows offset by sqrt(3)/2 * diameter
        
        rows = int(math.sqrt(n * 2 / math.sqrt(3))) + 2
        
        count = 0
        row = 0
        
        while count < n and row < rows:
            # Calculate y position for this row
            y = (row + 0.5) * (square_size / (rows + 1))
            
            # Number of circles in this row
            if row % 2 == 0:
                cols = int(math.sqrt(n)) + 1
            else:
                cols = int(math.sqrt(n))
            
            spacing_x = square_size / (cols + 1)
            
            for col in range(cols):
                if count >= n:
                    break
                
                if row % 2 == 0:
                    x = spacing_x * (col + 1)
                else:
                    x = spacing_x * (col + 1) + spacing_x / 2
                
                # Calculate maximum radius for this position
                r = max_circle_radius(x, y, circles, square_size)
                
                if r > 0:
                    circles.append((x, y, r))
                    count += 1
            
            row += 1
        
        return circles

    def optimize_placement(n, square_size=1.0):
        """
        Optimize circle placement using uniform tiling with radius maximization.
        """
        circles = []
        
        # First, try hexagonal packing for high initial density
        hex_circles = hexagonal_packing(n, square_size)
        if len(hex_circles) == n:
            # Ensure maximum radii for hex layout with stronger refinement
            hex_refined = refine_circles(hex_circles, square_size, iterations=20)
            return hex_refined
        
        # Fallback to uniform grid placement
        grid_circles = uniform_tiling_circles(n, square_size)
        if len(grid_circles) == n:
            return grid_circles
        
        # If uniform tiling didn't work perfectly, use adaptive approach
        # Calculate optimal radius based on density
        area_per_circle = (square_size * square_size) / n
        estimated_radius = math.sqrt(area_per_circle / math.pi) * 0.9  # Conservative estimate
        
        # Create grid with optimal spacing
        spacing = estimated_radius * 2.1  # Include gap
        
        cols = int(square_size / spacing)
        rows = int(square_size / spacing)
        
        actual_spacing_x = square_size / (cols + 1)
        actual_spacing_y = square_size / (rows + 1)
        
        count = 0
        for row in range(rows):
            for col in range(cols):
                if count >= n:
                    break
                
                x = actual_spacing_x * (col + 1)
                y = actual_spacing_y * (row + 1)
                
                # Calculate maximum possible radius
                r = max_circle_radius(x, y, circles, square_size)
                
                if r > 0:
                    circles.append((x, y, r))
                    count += 1
            
            if count >= n:
                break
        
        # If we still need more circles, use remaining space
        remaining = n - len(circles)
        if remaining > 0:
            # Place remaining circles in remaining spaces
            for i in range(remaining):
                # Try different positions systematically
                best_r = 0
                best_pos = (0.5, 0.5)
                
                # Fine grid search (increased resolution)
                grid_points = 100
                for gx in range(1, grid_points):
                    for gy in range(1, grid_points):
                        x = gx / grid_points
                        y = gy / grid_points
                        
                        r = max_circle_radius(x, y, circles, square_size)
                        if r > best_r:
                            best_r = r
                            best_pos = (x, y)
                
                if best_r > 0:
                    circles.append((best_pos[0], best_pos[1], best_r))
        
        return circles

    def refine_circles(circles, square_size, iterations=80, perturb_interval=3):
        """
        Iteratively grow each circle to its maximum radius under non-overlap constraints.
        Includes randomized update order, periodic micro-perturbation to escape
        local minima, and a final local-center-perturbation pass for densification.
        """
        for it in range(iterations):
            # randomize update order to avoid sweep-order bias
            indices = list(range(len(circles)))
            random.shuffle(indices)
            for i in indices:
                x, y, _ = circles[i]
                # Compute maximal feasible radius here, skipping self
                r = max_circle_radius(x, y, circles, square_size, skip_idx=i)
                circles[i] = (x, y, r)
            # Periodic micro-perturbation: jiggle a few circles
            if it % perturb_interval == 0 and len(circles) > 0:
                subset = random.sample(indices, min(5, len(circles)))
                for j in subset:
                    x0, y0, r0 = circles[j]
                    dx = random.uniform(-0.03, 0.03)
                    dy = random.uniform(-0.03, 0.03)
                    nx = min(max(x0 + dx, 0), square_size)
                    ny = min(max(y0 + dy, 0), square_size)
                    # Compute maximal radius skipping self
                    nr = max_circle_radius(nx, ny, circles, square_size, skip_idx=j)
                    if nr > r0:
                        circles[j] = (nx, ny, nr)
        # Full local center-perturbation phase for final densification
        for i in range(len(circles)):
            x, y, r = circles[i]
            best_x, best_y, best_r = x, y, r
            delta = 0.1
            for _ in range(20):
                dx = random.uniform(-delta, delta)
                dy = random.uniform(-delta, delta)
                nx = min(max(x + dx, 0), square_size)
                ny = min(max(y + dy, 0), square_size)
                # Compute maximal radius skipping self
                nr = max_circle_radius(nx, ny, circles, square_size, skip_idx=i)
                if nr > best_r:
                    best_x, best_y, best_r = nx, ny, nr
                else:
                    delta *= 0.9
            circles[i] = (best_x, best_y, best_r)
        
        # Physics-inspired soft relaxation to escape persistent overlaps
        for i in range(len(circles)):
            x, y, r = circles[i]
            fx, fy = 0.0, 0.0
            for j, (xj, yj, rj) in enumerate(circles):
                if i == j:
                    continue
                dx = x - xj
                dy = y - yj
                d = (dx*dx + dy*dy) ** 0.5
                overlap = (r + rj) - d
                if overlap > 0 and d > 1e-8:
                    fx += dx / d * overlap
                    fy += dy / d * overlap
            # Nudge the center by 10% of the computed net “repulsive” force
            nx = min(max(x + 0.1 * fx, 0), square_size)
            ny = min(max(y + 0.1 * fy, 0), square_size)
            nr = max_circle_radius(nx, ny, circles, square_size, skip_idx=i)
            circles[i] = (nx, ny, nr)
        return circles

    def multi_start_optimize(n, square_size, starts=None):
        """
        Parallel multi-start global → local optimization using ThreadPoolExecutor.
        Number of starts adapts to problem size: max(100, 10*n).
        """
        if starts is None:
            if n <= 50:
                starts = max(200, n * 20)
            else:
                starts = max(100, n * 10)
        # precompute hexagonal‐packing baseline
        hex_circ = hexagonal_packing(n, square_size)
        hex_sum = sum(r for _, _, r in hex_circ)
        best_conf = None
        best_sum = 0.0

        # single trial: seed → refine → score
        def single_run(_):
            conf0 = optimize_placement(n, square_size)
            conf1 = refine_circles(conf0, square_size, iterations=40)
            s1 = sum(r for _, _, r in conf1)
            return s1, conf1

        # dispatch trials in parallel
        with ThreadPoolExecutor() as executor:
            for score, conf in executor.map(single_run, range(starts)):
                if score > best_sum:
                    best_sum, best_conf = score, conf.copy()
                # early exit if near the hex-baseline
                if best_sum >= hex_sum * 0.995:
                    break

        return best_conf

    # Use multi-start global → local optimization (adaptive number of starts)
    circles = multi_start_optimize(n, square_size)

    # Quick 2-cluster remove-and-reinsert densification (extended iterations)
    for _ in range(8):
        # remove the two smallest circles to create a larger gap
        smallest = sorted(range(len(circles)), key=lambda i: circles[i][2])[:2]
        removed = [circles[i] for i in smallest]
        # pop in reverse order to keep indices valid
        for i in sorted(smallest, reverse=True):
            circles.pop(i)
        # refine the remaining configuration briefly
        circles = refine_circles(circles, square_size, iterations=8)
        # reinsert each removed circle with more sampling
        for x_old, y_old, _ in removed:
            best_r, best_pos = 0.0, (x_old, y_old)
            for _ in range(500):
                x = random.uniform(0, square_size)
                y = random.uniform(0, square_size)
                r = max_circle_radius(x, y, circles, square_size)
                if r > best_r:
                    best_r, best_pos = r, (x, y)
            circles.append((best_pos[0], best_pos[1], best_r))
        # final local polish after reinsertion
        circles = refine_circles(circles, square_size, iterations=5)
    # end 2-cluster remove-and-reinsert densification

    # Calculate total radius
    total_radius = sum(circle[2] for circle in circles)
    
    return total_radius, circles