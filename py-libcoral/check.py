import time
from icecream import ic
import libcoral
import numpy as np

n = 10_000_000
dim = 10

rng = np.random.default_rng(212)
pts = rng.uniform(0, 10, size=(n, dim)).astype(np.float32)

coreset = libcoral.Coreset(1000, 16)
start = time.time()
coreset.fit(pts)
elapsed = time.time() - start
coreset_points, weights, radius = coreset.get_fit()

ic(coreset_points, weights, radius)
ic(elapsed)
