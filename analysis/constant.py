from __future__ import print_function, division, absolute_import

import numpy as np

class Constant(object):
    """This class contains physical constants"""

    def __init__(self):
        # Boltzmann constant: unit (J/K)
        self._kB = 1.38*1e-23
        # Vaccum permitivity: unit (F/m = C2/J/m)
        self._eps0 = 8.854*1e-12 * 4*np.pi

    @property
    def kB(self):
        return(self._kB)

    @property
    def eps0(self):
        return(self._eps0)

