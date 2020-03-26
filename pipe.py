import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as md
from dipole import Dipole


u = md.Universe('trj/md_300k.tpr',
                'trj/md_300k_100frame_pbc.xtc')
d = Dipole(u)
total_dipole_mat = d.total_dipole()

Mtot = np.loadtxt('trj/Mtot.xvg')

for i in range(3,4):
    plt.plot(total_dipole_mat[:,i])
    plt.plot(Mtot[:,i+1]*0.20819434, 'o')
plt.show()
