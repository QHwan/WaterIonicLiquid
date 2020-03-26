class Parameter(object):
    """Parameters used in molecular dynamics simulation"""

    def __init__(self):
        self._atom_type_vec = ['OW', 'HW1', 'HW2']
        # atomic charge: unit (e)
        self._charge_dict = {'OW': -0.8476,
                            'HW1': 0.4238,
                            'HW2': 0.4238}
        # LJ sigma: unit (nm)
        self._sigma_dict = {'OW': 0.316557,
                           'HW1': 0.,
                           'HW2': 0.}
        # LJ epsilon: unit (kJ/mol)
        self._epsilon_dict = {'OW': 0.65019,
                             'HW1': 0.,
                             'HW2': 0.}

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


