from __future__ import print_function, division, absolute_import

import networkx as nx
import numpy as np
import numba as nb
import MDAnalysis as md
import MDAnalysis.analysis.distances as mdanadist

from util import check_pbc_vec


class HydrogenBond(object):
    """Hydrogen Bonds of system."""

    def __init__(self, universe):
        """
        Parameters
        ----------
        universe : :obj:'MDAnalysis.core.universe.Universe'

        """
        self._universe = universe

        self._r_c = 3.5
        self._cos_ang_c = np.cos(np.pi/6)

    @property
    def universe(self):
        return self._universe

    @property
    def r_c(self):
        return self._r_c

    @property
    def cos_ang_c(self):
        return self._cos_ang_c


def hydrogen_bond_pair(HB, ts):
    """
    Parameters
    ----------
    HB : :obj: 'WaterIonicLiquid.hydrogen_bond.HydrogenBond'
    ts : :obj: 'MDAnalysis.coordinates.base'

    Returns
    -------
    hb_pair : int[:,:], shape = (n_hb, 3)
        column -> donor_idx, hydrogen_idx, acceptor_idx
    """
    box = ts.dimensions
    sol = HB.universe.select_atoms('name OW or name HW1 or name HW2')
    x_sol = sol.positions
    x_ow = x_sol[sol.names == "OW"]

    r_pair = mdanadist.distance_array(x_ow, x_ow, box=box)

    idx_hb_r_c = np.array(np.where((r_pair <= HB.r_c) & (r_pair > 0))).T   
    
    hb_pair_list = []
    for idx_hb in idx_hb_r_c:
        idx_i, idx_j = idx_hb*3

        pbc_x_i = check_pbc_vec(x_sol[idx_j], x_sol[idx_i], box)
        pbc_x_j = check_pbc_vec(x_sol[idx_i], x_sol[idx_j], box)

        ang = hda_ang(x_sol[idx_i+1], x_sol[idx_i], pbc_x_j)
        if ang > HB.cos_ang_c:
            hb_pair_list.append([idx_i, idx_i+1, idx_j])
            continue

        ang = hda_ang(x_sol[idx_i+2], x_sol[idx_i], pbc_x_j)
        if ang > HB.cos_ang_c:
            hb_pair_list.append([idx_i, idx_i+2, idx_j])
            continue

        ang = hda_ang(x_sol[idx_j+1], x_sol[idx_j], pbc_x_i)
        if ang > HB.cos_ang_c:
            hb_pair_list.append([idx_j, idx_j+1, idx_i])
            continue

        ang = hda_ang(x_sol[idx_j+2], x_sol[idx_j], pbc_x_i)
        if ang > HB.cos_ang_c:
            hb_pair_list.append([idx_j, idx_j+2, idx_i])
            continue
               
    hb_pair = np.array(hb_pair_list).astype(int)
    return(hb_pair)


def hydrogen_bond_graph(HB, hb_pair, kind='undirected'):
    """
    Parameters
    ----------
    HB : :obj: 'WaterIonicLiquid.hydrogen_bond.HydrogenBond'
    hb_pair : int[:,:], shape = (n_hb, 3)
        columns -> donor_idx, hydrogen_idx, acceptor_idx
    kind : str, (undirected)
        kind of graph structure
    """
    ow = HB.universe.select_atoms('name OW')
    nodes = np.array(range(len(ow)))
    edges = (np.array([hb_pair[:,0], hb_pair[:,2]])/3).T.astype(int)
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    return(G)


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


        