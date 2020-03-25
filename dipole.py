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
