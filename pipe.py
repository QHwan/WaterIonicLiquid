import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as md
import networkx as nx

from hydrogen_bond import hydrogen_bond_graph, hydrogen_bond_pair, HydrogenBond

u = md.Universe('trj/md_300k.tpr',
                'trj/md_300k_100frame.xtc')

hb = HydrogenBond(u)
for i in range(1):
    ts = u.trajectory[i]
    _, hb_pair = hydrogen_bond_pair(hb, ts)

    hb_graph = hydrogen_bond_graph(hb, hb_pair, kind='directed')

    nx.draw_networkx(hb_graph, with_labels=False, node_size=20, linwidths=0.2, alpha=0.5)
    plt.show()

