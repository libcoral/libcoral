import time
from icecream import ic
import libcoral
import numpy as np
from scipy.spatial.distance import pdist

n = 10000
dim = 10

rng = np.random.default_rng(212)
pts = rng.uniform(0, 10, size=(n, dim)).astype(np.float32)
print("Generated points")

diversity = libcoral.DiversityMaximization(
    10,
    "remote-clique"
)
start = time.time()
diversity.fit(pts)
elapsed = time.time() - start

sol_idxs = diversity.solution_indices()

sol = pts[sol_idxs]

ic(elapsed, diversity.cost(sol))
