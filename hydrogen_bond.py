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

        self._r_owow_c = 3.5
        self._r_naow_c = 3.2
        self._r_hwcl_c = 3
        self._r_nacl_c = 3.6
        self._cos_ang_c = np.cos(np.pi/6)

    @property
    def universe(self):
        return self._universe

    @property
    def r_owow_c(self):
        return self._r_owow_c

    @property
    def r_naow_c(self):
        return self._r_naow_c
    
    @property
    def r_hwcl_c(self):
        return self._r_hwcl_c

    @property
    def r_nacl_c(self):
        return self._r_nacl_c

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
    r_pair : float[:,:], shape = (n_mol, n_mol)
    hb_pair : int[:,:], shape = (n_hb, 3)
        column -> donor_idx, hydrogen_idx, acceptor_idx
    """
    box = ts.dimensions

    sol = HB.universe.select_atoms('name OW or name HW1 or name HW2')
    na = HB.universe.select_atoms('name NA')
    cl = HB.universe.select_atoms('name CL')

    x_sol = sol.positions
    x_ow = x_sol[sol.names == "OW"]
    x_hw1 = x_sol[sol.names == "HW1"]
    x_hw2 = x_sol[sol.names == "HW2"]
    x_na = na.positions
    x_cl = cl.positions

    n_ow = len(x_ow)
    n_na = len(x_na)
    n_cl = len(x_cl)

    # HB - owow
    r_pair = mdanadist.distance_array(x_ow, x_ow, box=box)

    idx_hb_r_c = np.array(np.where((r_pair <= HB.r_owow_c) & (r_pair > 0))).T   
    
    hb_pair_list = []
    for idx_hb in idx_hb_r_c:
        idx_i, idx_j = idx_hb*3
        if idx_i > idx_j:
            continue

        pbc_x_i = check_pbc_vec(x_sol[idx_j], x_sol[idx_i], box)
        pbc_x_j = check_pbc_vec(x_sol[idx_i], x_sol[idx_j], box)

        ang = hda_ang(x_sol[idx_i+1], x_sol[idx_i], pbc_x_j)
        if ang > HB.cos_ang_c:
            hb_pair_list.append([int(idx_i/3), int(idx_j/3)])
            continue

        ang = hda_ang(x_sol[idx_i+2], x_sol[idx_i], pbc_x_j)
        if ang > HB.cos_ang_c:
            hb_pair_list.append([int(idx_i/3), int(idx_j/3)])
            continue

        ang = hda_ang(x_sol[idx_j+1], x_sol[idx_j], pbc_x_i)
        if ang > HB.cos_ang_c:
            hb_pair_list.append([int(idx_j/3), int(idx_i/3)])
            continue

        ang = hda_ang(x_sol[idx_j+2], x_sol[idx_j], pbc_x_i)
        if ang > HB.cos_ang_c:
            hb_pair_list.append([int(idx_j/3), int(idx_i/3)])
            continue

    # HB - naow
    r_pair = mdanadist.distance_array(x_na, x_ow, box=box)
    idx_hb_r_c = np.array(np.where((r_pair <= HB.r_naow_c) & (r_pair > 0))).T 
    for idx_hb in idx_hb_r_c:
        idx_na, idx_ow = idx_hb
        hb_pair_list.append([n_ow+idx_na, idx_ow]) 

    # HB - hwcl
    r_pair1 = mdanadist.distance_array(x_hw1, x_cl, box=box)
    r_pair2 = mdanadist.distance_array(x_hw2, x_cl, box=box) 
    idx_hb_r_c = np.array(np.where(((r_pair1 <= HB.r_hwcl_c) | (r_pair2 <= HB.r_hwcl_c)) &
                                   (r_pair1 > 0) &
                                   (r_pair2 > 0))).T 
    for idx_hb in idx_hb_r_c:
        idx_ow, idx_cl = idx_hb
        hb_pair_list.append([idx_ow, n_ow+n_na+idx_cl]) 

    # HB - nacl
    r_pair = mdanadist.distance_array(x_na, x_cl, box=box)
    idx_hb_r_c = np.array(np.where((r_pair <= HB.r_nacl_c) & 
                                   (r_pair > 0))).T 
    for idx_hb in idx_hb_r_c:
        idx_na, idx_cl = idx_hb
        hb_pair_list.append([n_ow+idx_na, n_ow+n_na+idx_cl])
              
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
    na = HB.universe.select_atoms('name NA')
    cl = HB.universe.select_atoms('name CL')
    
    n_ow = len(ow)
    n_na = len(na)
    n_cl = len(cl)

    edges = hb_pair

    if kind.lower() == 'undirected':
        G = nx.Graph()
    elif kind.lower() == 'directed':
        G = nx.DiGraph()

    #G.add_nodes_from(nodes)
    for i in range(n_ow):
        G.add_node(i, name='OW')
    for i in range(n_ow, n_ow+n_na):
        G.add_node(i, name='NA')
    for i in range(n_ow+n_na, n_ow+n_na+n_cl):
        G.add_node(i, name='CL')

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


        