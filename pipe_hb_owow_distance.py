import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as md
import MDAnalysis.analysis.distances as mdanadist
import seaborn as sns
import networkx as nx
import networkx.algorithms as nxal
from tqdm import tqdm

from hydrogen_bond import hydrogen_bond_graph, hydrogen_bond_pair, HydrogenBond

u = md.Universe('trj/md_0m.tpr',
                'trj/md_0m_1000frame.xtc')

ow = u.select_atoms('name OW')

n_ow = len(ow)


hb = HydrogenBond(u)

dist_path_list = []
for i, ts in tqdm(enumerate(u.trajectory), total=len(u.trajectory)):
    if i == 10:
        break
    
    hb_pair = hydrogen_bond_pair(hb, ts)
    G = hydrogen_bond_graph(hb, hb_pair, kind='directed')

    x_ow = ow.positions
    box = ts.dimensions
    r_owow_pair = mdanadist.distance_array(x_ow, x_ow, box=box) 

    for i in range(n_ow):
        for j in range(i+1, n_ow):
            dist = r_owow_pair[i, j]
            if dist > 10:
                continue

            try:
                path1 = nxal.shortest_paths.shortest_path(G, i, j)
                path2 = nxal.shortest_paths.shortest_path(G, j, i)

            except nx.exception.NetworkXNoPath as e:
                continue

            len_path = min(len(path1), len(path2))

            dist_path_list.append([dist, len_path])

dist_path_mat = np.array(dist_path_list)
np.savez('dist_path_0m.npz', dist_path=dist_path_mat)

plt.scatter(dist_path_mat[:,0], dist_path_mat[:,1])
plt.show()
            


