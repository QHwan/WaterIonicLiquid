from __future__ import print_function, division, absolute_import

import numpy as np


def sig_matrix(sig1, sig2):
    sig_mat = np.zeros((len(sig1), len(sig2)))
    for i in range(len(sig1)):
        for j in range(len(sig2)):
            sig_mat[i,j] = (sig1[i]+sig2[j])*0.5
    return(sig_mat)

def eps_matrix(eps1, eps2):
    return(np.sqrt(np.outer(eps1, eps2)))

def q_matrix(q1, q2):
    return(np.outer(q1, q2))


class Parameter(object):
    """Parameters used in molecular dynamics simulation"""

    def __init__(self):
        self._atom_type_vec = ['OW', 'HW1', 'HW2', 'NA', 'CL']
        # atomic charge: unit (e)
        self._charge_dict = {'OW': -0.8476,
                            'HW1': 0.4238,
                            'HW2': 0.4238,
                            'NA' : 1.0,
                            'CL' : -1.0}
        # LJ sigma: unit (angstrom)
        self._sigma_dict = {'OW': 0.316557 * 10,
                           'HW1': 0.,
                           'HW2': 0.,
                           'NA': 0.2584 * 10,
                           'CL': 0.4401 * 10}
        # LJ epsilon: unit (kJ/mol)
        self._epsilon_dict = {'OW': 0.65019,
                             'HW1': 0.,
                             'HW2': 0.,
                             'NA': 0.4184,
                             'CL': 0.4184}

        # check missing parameters
        for atom_type in self._atom_type_vec:
            if atom_type not in self._charge_dict:
                raise KeyError('You miss charge of atom.')
            if atom_type not in self._sigma_dict:
                raise KeyError('You miss LJ sigma parameter of atom.')
            if atom_type not in self._epsilon_dict:
                raise KeyError('You miss LJ epsilon parameter of atom.')

    @property
    def charge_dict(self):
        return self._charge_dict

    @property
    def sigma_dict(self):
        return self._sigma_dict

    @property
    def epsilon_dict(self):
        return self._epsilon_dict


