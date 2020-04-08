import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as md
import MDAnalysis.analysis.distances as mdanadist
import seaborn as sns
import networkx as nx
import networkx.algorithms as nxal
from tqdm import tqdm

from hydrogen_bond import hydrogen_bond_graph, hydrogen_bond_pair, HydrogenBond

u = md.Universe('trj/md_2m.tpr',
                'trj/md_2m_1000frame.xtc')

ow = u.select_atoms('name OW')
na = u.select_atoms('name NA')
cl = u.select_atoms('name CL')

n_ow = len(ow)
n_na = len(na)
n_cl = len(cl)


hb = HydrogenBond(u)

dist_path_list = []
for i, ts in tqdm(enumerate(u.trajectory), total=len(u.trajectory)):
    if i == 100:
        break
    
    hb_pair = hydrogen_bond_pair(hb, ts)
    G = hydrogen_bond_graph(hb, hb_pair, kind='directed')

    x_na = na.positions
    x_cl = cl.positions
    box = ts.dimensions
    r_nacl_pair = mdanadist.distance_array(x_na, x_cl, box=box) 

    node_na = list(range(n_ow, n_ow+n_na))
    node_cl = list(range(n_ow+n_na, n_ow+n_na+n_cl))
    for idx_na in node_na:
        for idx_cl in node_cl:
            dist = r_nacl_pair[idx_na-n_ow, idx_cl-n_ow-n_na]
            if dist > 10.:
                continue

            try:
                path = nxal.shortest_paths.shortest_path(G, idx_na, idx_cl)
            
            except nx.exception.NetworkXNoPath as e:
                continue

            len_path = len(path)

            dist_path_list.append([dist, len_path])

dist_path_mat = np.array(dist_path_list)
np.savez('dist_path_2m.npz', dist_path=dist_path_mat)

plt.scatter(dist_path_mat[:,0], dist_path_mat[:,1])
plt.show()
            


