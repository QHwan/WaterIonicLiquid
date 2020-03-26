from __future__ import print_function, division, absolute_import

class Constant(object):
    """This class contains physical constants"""

    def __init__(self):
        # Boltzmann constant: unit (J/K)
        self._kB = 1.38*1e-23

    @property
    def kB(self):
        return self._kB

