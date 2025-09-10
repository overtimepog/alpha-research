import numpy as np

def jittered_grid_points(n, seed=0):
    """
    Place points on a near-uniform jittered grid inside [0,1]^2.
    n = m^2 or round to closest square. Good simple baseline.
    """
    rng = np.random.default_rng(seed)
    m = int(round(np.sqrt(n)))
    m = max(m, 2)
    xs = (np.arange(m) + 0.5) / m
    ys = (np.arange(m) + 0.5) / m
    X, Y = np.meshgrid(xs, ys)
    P = np.c_[X.ravel(), Y.ravel()]
    # jitter slightly to avoid collinearities and enlarge min triangles
    P += rng.uniform(-0.15/m, 0.15/m, size=P.shape)
    P = np.clip(P, 0.0, 1.0)
    return P[:n]

def hex_lattice_points(n, seed=0):
    """
    Approximate hexagonal packing clipped to the unit square; then jitter.
    Often gives slightly larger minimal triangles than a plain grid.
    """
    rng = np.random.default_rng(seed)
    a = 1.0 / np.sqrt(n)  # target spacing scale
    pts = []
    y = a/2
    row = 0
    while y < 1.0:
        x0 = (a/2) if (row % 2 == 1) else a
        x = x0
        while x < 1.0:
            pts.append([x, y])
            x += a
        y += np.sqrt(3)/2 * a
        row += 1
    P = np.array(pts, dtype=float)
    if len(P) < n:
        # fall back: add random points if packing is short
        extra = rng.uniform(0, 1, size=(n - len(P), 2))
        P = np.vstack([P, extra])
    P = P[:n]
    # small jitter
    P += rng.uniform(-0.1*a, 0.1*a, size=P.shape)
    P = np.clip(P, 0.0, 1.0)
    return P

def main():
    n = 30
    pts = hex_lattice_points(n, seed=42)
    print(f"n={n}, points={len(pts)}")
    return pts

if __name__ == "__main__":
    points = main()

# Ensure compatibility with evaluators that expect a global variable
try:
    points  # type: ignore[name-defined]
except NameError:
    points = main()
