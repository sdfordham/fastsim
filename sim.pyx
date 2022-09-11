# cython: language_level=3, boundscheck=False, wraparound=False
cimport cython
cimport numpy as np
import numpy as np


def _jaccard(np.int64_t[:] a, np.int64_t[:] b):
    cdef:
        Py_ssize_t col_count = a.shape[0]
        float match = 0
        float mismatch = 0
    for i in range(col_count):
        if a[i] == 1 and b[i] == 1:
            match += 1
        elif a[i] != 0 and b[i] != 0:
            mismatch += 1
    return match / (match + mismatch)


def _dice(np.int64_t[:] a, np.int64_t[:] b):
    cdef:
        Py_ssize_t col_count = a.shape[0]
        float match = 0
        float mismatch = 0
    for i in range(col_count):
        if a[i] == 1 and b[i] == 1:
            match += 1
        elif a[i] != 0 and b[i] != 0:
            mismatch += 1
    return 2. * match / (2. * match + mismatch)


def _russellrao(np.int64_t[:] a, np.int64_t[:] b):
    cdef:
        Py_ssize_t col_count = a.shape[0]
        float match = 0
        float mismatch = 0
    for i in range(col_count):
        if a[i] == 1 and b[i] == 1:
            match += 1
    return (col_count - match) / col_count


metrics = {
    'jaccard': _jaccard,
    'dice': _dice,
    'russellrao': _russellrao
}


def similarity_matrix(np.int64_t[:, :] rows, metric='jaccard'):
    metric_fn = metrics[metric]
    cdef:
        Py_ssize_t row_count = rows.shape[0]
        np.float64_t[:, :] result = np.zeros((row_count, row_count), dtype=np.float64)
    for i in range(row_count):
        for j in range(i, row_count):
            if i != j:
                d = metric_fn(rows[i], rows[j])
                result[i, j] = 1. - d
                result[j, i] = 1. - d
    return np.asarray(result)