from __future__ import print_function, division, absolute_import

import numpy as np
import numba as nb


@nb.njit(fastmath=True)
def check_pbc(ref_x1, x2, box):
    pbc_x2 = np.copy(x2)
    for i in range(3):
        if ref_x1[i] - x2[0,i] > box[i]/2:
            for j in range(len(x2)):
                pbc_x2[j,i] += box[i]
        elif x2[0,i] - ref_x1[i] > box[i]/2:
            for j in range(len(x2)):
                pbc_x2[j,i] -= box[i]
    return(pbc_x2)


@nb.njit(fastmath=True)
def distance(x1, x2):
    d = 0
    for i in range(len(x1)):
        d += (x2[i] - x1[i])**2
    return(d**0.5)


@nb.njit(fastmath=True)
def distance_array(x1, x2):
    d = np.zeros((len(x1), len(x2)))
    for i in range(len(x1)):
        for j in range(len(x2)):
            d[i,j] = distance(x1[i], x2[j])
    return(d)


def running_mean(x):
    """Calculate running mean of x

    Parameters
    ----------
    x : float[:]

    Returns
    -------
    run_x : float[:], shape_like x
    """
    run_x = np.zeros_like(x)
    for i in range(len(x)):
        if i == 0:
            avg = x[i]
        else:
            avg *= i
            avg += x[i]
            avg /= (i+1)
        run_x[i] = avg
    return(run_x)
