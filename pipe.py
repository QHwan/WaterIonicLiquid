import MDAnalysis as md
from dipole import Dipole

u = md.Universe('trj/md_300k.tpr',
                'trj/md_300k_100frame_pbc.xtc')
d = Dipole(u)
