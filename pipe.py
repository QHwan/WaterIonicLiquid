import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as md

from hydrogen_bond import hydrogen_bond_matrix

u = md.Universe('trj/md_300k.tpr',
                'trj/md_300k_100frame.xtc')

import time

ow = u.select_atoms('name OW')
atom = u.select_atoms('all')
for i in range(10):
    start_time = time.time()

    ts = u.trajectory[i]
    hydrogen_bond_matrix(atom.positions, ow.positions, ts.dimensions)
    print(time.time() - start_time)


