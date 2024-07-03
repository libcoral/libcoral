import time
from icecream import ic
import libcoral
import numpy as np
import logging
logging.basicConfig(level=logging.DEBUG)

n = 5000
dim = 10

rng = np.random.default_rng(212)
pts = rng.uniform(0, 10, size=(n, dim)).astype(np.float32)
print("Generated points")

diversity = libcoral.DiversityMaximization(
    5,
    "remote-clique",
    coreset_size=10,
    num_threads=2,
)
start = time.time()
diversity.fit(pts)
elapsed = time.time() - start

sol_idxs = diversity.solution_indices()

sol = pts[sol_idxs]

ic(elapsed, diversity.cost(sol))
