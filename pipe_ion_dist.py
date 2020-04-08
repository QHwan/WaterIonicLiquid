import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as md
import MDAnalysis.analysis.distances as mdanadist
import seaborn as sns
import networkx as nx
from tqdm import tqdm

from hydrogen_bond import hydrogen_bond_graph, hydrogen_bond_pair, HydrogenBond

u = md.Universe('trj/md_2m.tpr',
                'trj/md_2m_1000frame.xtc')

hb = HydrogenBond(u)

for i, ts in tqdm(enumerate(u.trajectory), total=len(u.trajectory)):
    hb_pair = hydrogen_bond_pair(hb, ts)
    G = hydrogen_bond_graph(hb, hb_pair, kind='directed')
    break

node_list = list(G.nodes(data=True))
node_color = []
for node in node_list:
    if node[1]['name'] == 'OW':
        node_color.append('red')
    elif node[1]['name'] == 'NA':
        node_color.append('blue')
    elif node[1]['name'] == 'CL':
        node_color.append('green')

#layout = nx.fruchterman_reingold_layout(G)
layout = nx.kamada_kawai_layout(G)

nx.draw_networkx(G, pos=layout, with_labels=False, node_size=10, node_color=node_color)
plt.show()

