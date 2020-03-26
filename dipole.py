from __future__ import print_function, division, absolute_import

import numpy as np
from tqdm import tqdm
import MDAnalysis as md

from parameter import Parameter

class Dipole(object):
    """Molecular dipoles of system."""

    def __init__(self, universe):
        """

        Parameters
        ----------
        universe : :obj:'MDAnalysis.core.universe.Universe'

        """
        self._universe = universe
        self._num_frame = len(self._universe.trajectory)
        self._atom_vec = self._universe.select_atoms('all')
        self._num_atom = len(self._atom_vec)
        self._charge_vec = self._initialize_parameters(Parameter())


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
        total_dipole_mat : float[:,:], shape = (num_frame, 4)
            4 columns contain x-, y-, z- direction and total.
        """
        total_dipole_mat = np.zeros((self._num_frame, 4))
        for i, ts in tqdm(enumerate(self._universe.trajectory), total=self._num_frame):
            pos_atom_mat = self._atom_vec.positions
            total_dipole_mat[i,:3] = np.sum(pos_atom_mat * self._charge_vec.reshape(-1, 1), axis=0) # broadcasting along axis = 1
        total_dipole_mat[:,3] = np.linalg.norm(total_dipole_mat[:,:3], axis=1)
        return(total_dipole_mat)

            
