import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import MDAnalysis as md
import MDAnalysis.analysis.distances as mdanadist

from potential import potential_sol_sol

import time

@nb.njit(fastmath=True)
def potential_sol_sol_frame(x_sol, box):
    n_ow = int(len(x_sol)/3)
    p = 0
    for i in range(n_ow):
        for j in range(i+1, n_ow):
            p_lj, p_c = potential_sol_sol(x_sol[3*i:3*i+3], x_sol[3*j:3*j+3], box)
            p += p_lj + p_c
    return(p)

u = md.Universe('trj/md_0m.tpr',
                'trj/md_0m_1000frame.xtc')

ow = u.select_atoms('name OW')
sol = u.select_atoms('name OW or name HW1 or name HW2')

n_ow = len(ow)

for i in range(5):
    start_time = time.time()

    ts = u.trajectory[i]
    box = ts.dimensions
    x_sol = sol.positions

    p = potential_sol_sol_frame(x_sol, box)

    print(p)
    print(time.time() - start_time)



