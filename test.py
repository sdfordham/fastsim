from time import time

import numpy as np
from numpy.random import default_rng
from scipy.spatial.distance import jaccard as scipy_jaccard

from sim import sim_matrix


def timing(f: callable):
    def wrapped(*args, **kwargs):
        start = time()
        res = f(*args, **kwargs)
        end = time()
        print(f'{f.__name__} took {end - start:.4f} seconds.')
        return res
    return wrapped


def naive_jaccard(a, b):
    n, matched = len(a), 0
    for i in range(n):
        if a[i] == b[i]:
            matched += 1
    return matched / n


@timing
def basic(data, jaccard_fn: callable):
    similiarity_matrix = np.zeros((R, R), dtype=np.float64)
    for i in range(R):
        for j in range(i, R):
            if i != j:
                d = jaccard_fn(data[i], data[j])
                similiarity_matrix[i, j] = 1. - d
                similiarity_matrix[j, i] = 1. - d


@timing
def cython_version(data):
    similiarity_matrix = sim_matrix(data)


if __name__ == '__main__':
    R, C, N, P = 800, 100, 1, 0.5

    rng = default_rng()
    test_data = rng.binomial(N, P, size=(R, C))

    basic(test_data, jaccard_fn=naive_jaccard)
    basic(test_data, jaccard_fn=scipy_jaccard)
    cython_version(test_data)