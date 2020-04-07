from __future__ import print_function, division, absolute_import

import numpy as np
from numpy.core.umath_tests import inner1d
import numba as nb
from tqdm import tqdm
import MDAnalysis as md
import MDAnalysis.analysis.distances as mdanadist

from parameter import Parameter
from util import check_pbc, check_pbc_vec


class HydrogenBond(object):
    """Hydrogen Bonds of system."""

    def __init__(self, universe):
        """

        Parameters
        ----------
        universe : :obj:'MDAnalysis.core.universe.Universe'

        """
        self._universe = universe
        self._atom_vec = self._universe.select_atoms('all')
        self._ow = self._universe.select_atoms('name OW')
        self._atomname_vec = self._atom_vec.names

        self._n_frame = len(self._universe.trajectory)
        self._n_atom = len(self._atom_vec)

        self._r_cut = 3.5
        self._cos_ang_cut = np.cos(np.pi/6)


def hydrogen_bond_matrix(pos_mat, pos_ow_mat, box_vec):
    """
    Parameters
    ----------
    timestep : :obj:'MDAnalysis.coordinates.base'

    Returns
    -------
    hb_mat : int[:,:], shape = (n_mol, n_mol)
    """
    _r_cut = 3.5
    _cos_ang_cut = np.cos(np.pi/6)
    n_mol = len(pos_ow_mat)
    #pos_ow_mat = pos_mat[self._atomname_vec == 'OW']
    pos_mat3 = pos_mat.reshape((n_mol, -1, 3))

    dist_mat = mdanadist.distance_array(pos_ow_mat, pos_ow_mat, box=box_vec)
    
    idx_dist = np.array(np.where((dist_mat <= _r_cut) & (dist_mat > 0)))
    
    hb_pair_mat = []
    for _, idx in enumerate(idx_dist.T):
        idx_i, idx_j = idx

        ang = hda_ang(pos_mat3[idx_i, 1],
                      pos_mat3[idx_i, 0],
                      check_pbc_vec(pos_mat3[idx_i,0], pos_mat3[idx_j,0], box_vec))        
        if ang > _cos_ang_cut:
            hb_pair_mat.append([idx_i, idx_j])
            continue

        ang = hda_ang(pos_mat3[idx_i, 2],
                pos_mat3[idx_i, 0],
                check_pbc_vec(pos_mat3[idx_i,0], pos_mat3[idx_j,0], box_vec))        
        if ang > _cos_ang_cut:
            hb_pair_mat.append([idx_i, idx_j])
            continue

        ang = hda_ang(pos_mat3[idx_j, 1],
                        pos_mat3[idx_j, 0],
                        check_pbc_vec(pos_mat3[idx_j,0], pos_mat3[idx_i,0], box_vec))        
        if ang > _cos_ang_cut:
            hb_pair_mat.append([idx_i, idx_j])
            continue

        ang = hda_ang(pos_mat3[idx_j, 2],
                        pos_mat3[idx_j, 0],
                        check_pbc_vec(pos_mat3[idx_j,0], pos_mat3[idx_i,0], box_vec))        
        if ang > _cos_ang_cut:
            hb_pair_mat.append([idx_i, idx_j])
            continue
                

    hb_pair = np.array(hb_pair_mat)
    print(len(hb_pair))

@nb.jit(nopython=True)
def hda_ang(h, d, a):
    dh = np.zeros(3)
    da = np.zeros(3)
    for i in range(3):
        dh[i] = h[i] - d[i]
        da[i] = a[i] - d[i]
    return(dot(dh, da)/norm(dh)/norm(da))

@nb.jit(nopython=True)
def dot(a, b):
    n = 0
    for i in range(3):
        n += a[i]*b[i]
    return(n)

@nb.jit
def norm(a):
    n = 0
    for i in range(3):
        n += a[i]**2
    return(n**0.5)


        