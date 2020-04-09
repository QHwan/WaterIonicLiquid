from __future__ import print_function, division, absolute_import

import numpy as np
import numba as nb
from tqdm import tqdm
import MDAnalysis as md
import MDAnalysis.analysis.distances as mdanadist

<<<<<<< HEAD:analysis/potential.py
from .parameter import Parameter
from .util import check_pbc, distance_array





@nb.jit
=======
from parameter import Parameter

param = Parameter()

q_ow = param.charge_dict['OW']
q_hw = param.charge_dict['HW1']

sig_ow = param.sigma_dict['OW']
eps_ow = param.epsilon_dict['OW']


@nb.njit(fastmath=True)
>>>>>>> 2b9ec1787887c53336aafff14a66052b412e0682:potential.py
def lj(r, sig, eps):
        """Calculate lennard jones potential
        Parameters
        ----------
        r : float, unit -> angstrom
        sig : float, unit -> angstrom
        eps : float, unit -> kJ/mol
   
        Returns
        -------
        lj : float, unit -> kJ/mol
        """
        r6 = (sig/r)**6
        return(4*eps*r6*(r6-1))


<<<<<<< HEAD:analysis/potential.py

@nb.jit
def coul(r, q):
=======
@nb.njit(fastmath=True)
def coul(r, qi, qj):
>>>>>>> 2b9ec1787887c53336aafff14a66052b412e0682:potential.py
    """Calculate coulomb potential
    Parameters
    ----------
    r : float, unit -> angstrom
<<<<<<< HEAD:analysis/potential.py
    q: float, unit -> e*e
=======
    qi, qj : float, unit -> e
>>>>>>> 2b9ec1787887c53336aafff14a66052b412e0682:potential.py
    
    Returns
    -------
    c : float, unit -> kJ/mol
    """
<<<<<<< HEAD:analysis/potential.py
    return((q/r) * 138.935458 * 10)





@nb.jit
def potential_between_molecule(x1, x2, sig, eps, q, box):
    """ Calculate molecule-molecule potential

    Parameters
    -----------
    x1, x2 : float[:,:], shape -> (n_atoms, 3)
        position matrix of molecules
    sig : float[:,:], shape -> (n1_atoms, n2_atoms)
    eps : float[:,:], shape -> (n1_atoms, n2_atoms)
    q : float[:,:], shape -> (n1_atoms, n2_atoms)
=======
    return((qi*qj/r) * 138.935458 * 10)


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


@nb.njit(fastmath=True)
def potential_sol_sol(x1, x2, box):
    """ Calculate water-water potential.
    It can calculate SPC/E water now.
    Note: Further modification.

    Parameters
    -----------
    x1, x2 : float[3,3]
        position matrix of two water molecules
>>>>>>> 2b9ec1787887c53336aafff14a66052b412e0682:potential.py
    box : float[:]

    Returns
    --------
    p : float
    """
    pbc_x2 = check_pbc(x1[0], x2, box)
    r = distance_array(x1, pbc_x2) 

    if r[0,0] > 15:
        return(0, 0)

<<<<<<< HEAD:analysis/potential.py
    p_lj = 0
    p_c = 0
    for i in range(len(x1)):
        for j in range(len(x2)):
            if sig[i,j] != 0:
                p_lj += lj(r[i,j], sig[i,j], eps[i,j])
            if q[i,j] != 0:
                p_c += coul(r[i,j], q[i,j])

    return(p_lj, p_c)

=======
    p_lj = lj(r[0,0], sig_ow, eps_ow)
    p_c = (coul(r[0,0], q_ow, q_ow) + 
            coul(r[0,1], q_ow, q_hw) + 
            coul(r[0,2], q_ow, q_hw) + 
            coul(r[1,0], q_hw, q_ow) +
            coul(r[1,1], q_hw, q_hw) + 
            coul(r[1,2], q_hw, q_hw) + 
            coul(r[2,0], q_hw, q_ow) +
            coul(r[2,1], q_hw, q_hw) +
            coul(r[2,2], q_hw, q_hw)) 

    return(p_lj, p_c)
>>>>>>> 2b9ec1787887c53336aafff14a66052b412e0682:potential.py


class Potential(object):
    """Calculate potential of system.
    It can calculate potential of SPC/E system.
    Further modification is required.
    """

    def __init__(self, universe):
        """

        Parameters
        ----------
        universe : :obj:'MDAnalysis.core.universe.Universe'

        """
        self._universe = universe
        self._atom_vec = self._universe.select_atoms('all')

        self._num_frame = len(self._universe.trajectory)
        self._num_atom = len(self._atom_vec)

        self._param = Parameter()
        self._charge_vec, self._sigma_vec, self._epsilon_vec = self._initialize_parameters(self._param)


    def _initialize_parameters(self, param):
        """Initialize relevant parameters of atoms
        which cannot obtained from MDAnalysis module.
        Return flattend matrix: 2D -> 1D.

        Parameters
        ----------
        param : :obj:'parameter.Parameter'

        Returns
        -------
        charge_vec : float[:], shape = (self._num_atom * self._num_atom)
            q_i * q_j
        sigma_vec : float[:], shape = (self._num_atom * self._num_atom)
            (sigma_i+sigma_j)/2
        epsilon_vec : float[:], shape = (self._num_atom * self._num_atom)
            sqrt(eps_i*eps_j)
        """
        atom_name_vec = self._atom_vec.names
        charge_mat = np.array([[self._param.charge_dict[i]*self._param.charge_dict[j]
                                for j in atom_name_vec]
                                for i in atom_name_vec])
        sigma_mat = np.array([[(self._param.sigma_dict[i]+self._param.sigma_dict[j])*0.5
                                for j in atom_name_vec]
                                for i in atom_name_vec])
        sigma_mat *= 10
        epsilon_mat = np.array([[np.sqrt(self._param.epsilon_dict[i]*self._param.epsilon_dict[j])
                                for j in atom_name_vec]
                                for i in atom_name_vec])

        for i in range(int(self._num_atom/4)):
            for j in range(4):
                for k in range(4):
                    charge_mat[4*i+j, 4*i+k] = 0
        #np.fill_diagonal(charge_mat, 0)
        np.fill_diagonal(sigma_mat, 0)
        np.fill_diagonal(epsilon_mat, 0)
        return(charge_mat.ravel(), sigma_mat.ravel(), epsilon_mat.ravel())


    def potential_matrix(self):
        """Calculate molecular potential of nanodroplet.
        Returns
        -------
        lj_mat : float[:,:], shape = (self._num_frame, num_mol), unit = (kJ/mol)
        coul_mat : float[:,:], shape = (self._num_frame, num_mol), unit = (kJ/mol)
        """
        num_mol = int(self._num_atom/4)
        lj_mat = np.zeros((self._num_frame, num_mol))
        coul_mat = np.zeros_like(lj_mat)

        for i, ts in tqdm(enumerate(self._universe.trajectory), total=self._num_frame):
            dist_atom_vec = np.zeros((self._num_atom**2))
            dist_atom_vec = mdanadist.distance_array(self._atom_vec.positions, self._atom_vec.positions, box=ts.dimensions).ravel()

            lj_mat[i] += np.sum(np.sum(self._lennard_jones(dist_atom_vec), axis=1).reshape((-1,4)), axis=1)
            coul_mat[i] += np.sum(np.sum(self._coulomb(dist_atom_vec), axis=1).reshape((-1,4)), axis=1)

        return(lj_mat, coul_mat)



    def perturbed_potential_matrix(self, zeta=1e-5):
        """Calculate perturbed potential of nanodroplet.
        Returns
        -------
        dlj_mat : float[:,:], shape = (self._num_frame, num_mol), unit = (kJ/mol)
        dcoul_mat : float[:,:], shape = (self._num_frame, num_mol), unit = (kJ/mol)
        """
        num_mol = int(self._num_atom/4)
        extn_lj_mat = np.zeros((self._num_frame, num_mol))
        comp_lj_mat = np.zeros_like(extn_lj_mat)
        extn_coul_mat = np.zeros_like(extn_lj_mat)
        comp_coul_mat = np.zeros_like(extn_lj_mat)

        for i, ts in tqdm(enumerate(self._universe.trajectory), total=self._num_frame):
            box_vec = ts.dimensions[:3] 
            pos_atom_mat = self._atom_vec.positions
            pbc_pos_atom_mat = check_pbc(pos_atom_mat[0], pos_atom_mat, box_vec)

            com_vec = center_of_mass(pbc_pos_atom_mat[0::4])
            pbc_pos_atom_mat -= com_vec
            
            extn_pos_atom_mat = np.zeros_like(pos_atom_mat)
            comp_pos_atom_mat = np.zeros_like(pos_atom_mat)
            
            extn_pos_atom_mat[0::4] = pbc_pos_atom_mat[0::4]*(1+zeta)
            comp_pos_atom_mat[0::4] = pbc_pos_atom_mat[0::4]*(1)

            for j in range(1,4):
                extn_pos_atom_mat[j::4] = extn_pos_atom_mat[0::4] + (pbc_pos_atom_mat[j::4] - pbc_pos_atom_mat[0::4])
                comp_pos_atom_mat[j::4] = comp_pos_atom_mat[0::4] + (pbc_pos_atom_mat[j::4] - pbc_pos_atom_mat[0::4])


            extn_dist_atom_vec = distance_vector(extn_pos_atom_mat)
            comp_dist_atom_vec = distance_vector(comp_pos_atom_mat)

            extn_lj_mat[i] += np.sum(np.sum(self._lennard_jones(extn_dist_atom_vec), axis=1).reshape((-1,4)), axis=1)
            extn_coul_mat[i] += np.sum(np.sum(self._coulomb(extn_dist_atom_vec), axis=1).reshape((-1,4)), axis=1)
            comp_lj_mat[i] += np.sum(np.sum(self._lennard_jones(comp_dist_atom_vec), axis=1).reshape((-1,4)), axis=1)
            comp_coul_mat[i] += np.sum(np.sum(self._coulomb(comp_dist_atom_vec), axis=1).reshape((-1,4)), axis=1)

        return(extn_lj_mat - comp_lj_mat, extn_coul_mat - comp_coul_mat)


    def _lennard_jones(self, dist_vec):
        """Calculate lennard jones potential matrix of given distance matrix
        Parameters
        ----------
        dist_mat : float[:], shape = (self._num_atom * self._num_atom)
        
        Returns
        -------
        lj_mat : float[:,:], shape = (self._num_atom, self._num_atom)
        """
        lj_vec = np.zeros(self._num_atom**2)
        r6_vec = np.zeros_like(lj_vec)
        mask = np.where(self._epsilon_vec != 0)
        r6_vec[mask] = (self._sigma_vec[mask]/dist_vec[mask])**6
        lj_vec[mask] = 4*self._epsilon_vec[mask]*r6_vec[mask]*(r6_vec[mask]-1)
        return(lj_vec.reshape((self._num_atom, self._num_atom)))


    def _coulomb(self, dist_vec):
        """Calculate coulomb potential matrix of given distance matrix
        Parameters
        ----------
        dist_vec : float[:], shape = (self._num_atom * self._num_atom)
        
        Returns
        -------
        coul_mat : float[:,:], shape = (self._num_atom, self._num_atom)
        """
        coul_vec = np.zeros(self._num_atom**2)
        mask = np.where(self._charge_vec != 0)
        coul_vec[mask] = self._charge_vec[mask]/dist_vec[mask]
        coul_vec *= 138.935458 * 10 # 10 is angstrom -> nm conversion
        return(coul_vec.reshape((self._num_atom, self._num_atom)))

           


## Test Suite ##
if __name__ == "__main__":
    u = md.Universe('trj/md3.tpr', 'trj/md3.gro')
    pot = Potential(u)
    box_vec = u.trajectory[0].dimensions[:3] 
    pos_atom_mat = pot._atom_vec.positions
    pbc_pos_atom_mat = check_pbc(pos_atom_mat[0], pos_atom_mat, box_vec)
    dist_atom_vec = distance_vector(pbc_pos_atom_mat)

    lj_mat = pot._lennard_jones(dist_atom_vec)
    coul_mat = pot._coulomb(dist_atom_vec)

    print('#############################################')
    print('######## Unit Test: Potential Module ########')
    print('#############################################')
    print('\n')
    print('Potential calculated from WaterNanodroplet.potential module')
    print('-----------------------------------------------------------')
    print('LJ: {lj}, Coul: {coul}'.format(lj=np.sum(lj_mat)/2,
                                          coul=np.sum(coul_mat)/2))
    print('\n')
    print('Potential calculated from gromacs gmx energy')
    print('--------------------------------------------')
    print('LJ: {lj}, Coul: {coul}'.format(lj=3.9041, coul=-25.7737))
