import numpy as np
import MDAnalysis as md

class Dipole(object):
    """Molecular dipoles of system."""

    def __init__(self, universe):
        """
        Parameters
        ----------
            universe : :obj:'MDAnalysis.core.universe.Universe'
        """
        self.u = universe
        self.all_atom = self.u.select_atoms('all')
        print(self.all_atom.names)
