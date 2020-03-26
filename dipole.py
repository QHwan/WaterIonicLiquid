from __future__ import print_function, division, absolute_import

import numpy as np
from tqdm import tqdm
import MDAnalysis as md

from parameter import Parameter
from constant import Constant

import matplotlib.pyplot as plt

class Dipole(object):
    """Molecular dipoles of system."""

    def __init__(self, universe):
        """

        Parameters
        ----------
        universe : :obj:'MDAnalysis.core.universe.Universe'

        """
        self._universe = universe
        self._atom_vec = self._universe.select_atoms('all')
        self._charge_vec = self._initialize_parameters(Parameter())

        self._num_frame = len(self._universe.trajectory)
        self._num_atom = len(self._atom_vec)


    def _initialize_parameters(self, param):
        """Initialize relevant parameters of atoms
        which cannot obtained from MDAnalysis module.

        Parameters
        ----------
        param : :obj:'parameter.Parameter'

        Returns
        -------
        charge_vec : float[:], shape = (num_atom)

        """
        charge_dict = param.charge_dict
        atom_name_vec = self._atom_vec.names
        charge_vec = np.array([charge_dict[i] for i in atom_name_vec])
        return(charge_vec)


    def total_dipole(self):
        """Calculate total amount of dipole of system.
        
        Returns
        -------
        tot_dip_mat : float[:,:], shape = (num_frame, 4), unit = (eA)
            4 columns contain x-, y-, z- direction and total.
        """
        tot_dip_mat = np.zeros((self._num_frame, 4))
        for i, ts in tqdm(enumerate(self._universe.trajectory), total=self._num_frame):
            pos_atom_mat = self._atom_vec.positions
            tot_dip_mat[i,:3] = np.sum(pos_atom_mat * self._charge_vec.reshape(-1, 1), axis=0) # broadcasting along axis = 1
        tot_dip_mat[:,3] = np.linalg.norm(tot_dip_mat[:,:3], axis=1)
        return(tot_dip_mat)


    def static_dielectric_constant(self):
        """Calculate static dielectric constant (w=0)

        Returns
        -------
        dielec_const_vec : float[:], shape = (num_frame)
        """
        const = Constant()
        tot_dip_mat = self.total_dipole()
        run_avg_dip_vec = self._running_mean(np.sum(tot_dip_mat[:,:3], axis=1))
        avg_dip = np.mean(tot_dip_mat[:,3])
        sqr_dip_vec = tot_dip_mat[:,3]**2
        run_avg_sqr_dip_vec = self._running_mean(sqr_dip_vec)

        box_mat = np.array([ts.dimensions for ts in self._universe.trajectory])
        vol_vec = box_mat[:,0]*box_mat[:,1]*box_mat[:,2]

        dielec_const_vec = np.zeros(self._num_frame)
        dielec_const_vec.fill(1/3.)
        dielec_const_vec *= run_avg_sqr_dip_vec - run_avg_dip_vec**2
        dielec_const_vec /= vol_vec*const.kB*300
        dielec_const_vec /= 3.45*1e16
        dielec_const_vec += 1
        return(dielec_const_vec)

            
    def _running_mean(self, x):
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
        return run_x
