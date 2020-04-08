import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as md
import MDAnalysis.analysis.distances as mdanadist
import networkx as nx

from hydrogen_bond import hydrogen_bond_graph, hydrogen_bond_pair, HydrogenBond

u = md.Universe('trj/md_2m.tpr',
                'trj/md_2m_1000frame.xtc')

hb = HydrogenBond(u)
for i in range(1):
    ts = u.trajectory[i]
    r_pair = mdanadist.distance_array(x_ow, x_ow, box=box) 
    #_, hb_pair = hydrogen_bond_pair(hb, ts)

    #hb_graph = hydrogen_bond_graph(hb, hb_pair, kind='directed')

    nx.draw_networkx(hb_graph, with_labels=False, node_size=20, linwidths=0.2, alpha=0.5)
    plt.show()

