import numpy as np


rng = np.random.default_rng()


for i in range(10):
    h = rng.permutation(24)
    print(h)