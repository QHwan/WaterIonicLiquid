import argparse
import MDAnalysis as md
import numpy as np
import numba as nb
import MDAnalysis.analysis.distances as mdanadist
from tqdm import tqdm
from collections import Counter


def get_status(c):
    if c[0] == 4:
        return 0
    elif c[0] == 3:
        if c[1] == 1:
            return 1
        elif c[2] == 1:
            return 2
    elif c[0] == 2:
        if c[1] == 2:
            return 3
        elif c[2] == 2:
            return 4
        elif c[1] == 1 and c[2] == 1:
            return 5
    elif c[0] == 1:
        if c[1] == 2 and c[2] == 1:
            return 6
        elif c[1] == 1 and c[2] == 2:
            return 7
    elif c[0] == 0 and c[1] == 2 and c[2] == 2:
        return 8
    return -1
    

parser = argparse.ArgumentParser()
parser.add_argument('--s', type=str, help='topology file')
parser.add_argument('--f', type=str, help='trajectory file')
parser.add_argument('--o', type=str, help='output file')

args = parser.parse_args()

u = md.Universe(args.s, args.f)


# atom selection
ow = u.select_atoms("name OW")
ca = u.select_atoms("name NA")
an = u.select_atoms("name CL")

n_ow = len(ow)
n_ca = len(ca)
n_an = len(an)

idx_ow_i = 0
idx_ow_f = n_ow-1
idx_ca_i = idx_ow_f+1
idx_ca_f = idx_ca_i+n_ca-1
idx_an_i = idx_ca_f+1
idx_an_f = idx_an_i+n_an-1

neighbor_mat = []
status_mat = np.zeros((len(u.trajectory), n_ow), dtype=int)

idx = 0

dist_ow_ow_mat = np.zeros((n_ow, n_ow))
dist_ow_ca_mat = np.zeros((n_ow, n_ca))
dist_ow_an_mat = np.zeros((n_ow, n_an))

for i, ts in tqdm(enumerate(u.trajectory), total=len(u.trajectory)):
    box = ts.dimensions

    pos_ow = ow.positions
    pos_ca = ca.positions
    pos_an = an.positions

    mdanadist.distance_array(pos_ow, pos_ow, box, 
                             result=dist_ow_ow_mat,
                             backend='OpenMP')
    mdanadist.distance_array(pos_ow, pos_ca, box, 
                             result=dist_ow_ca_mat,
                             backend='OpenMP')
    mdanadist.distance_array(pos_ow, pos_an, box, 
                             result=dist_ow_an_mat,
                             backend='OpenMP')

    
    dist_ow_mat = np.concatenate((dist_ow_ow_mat,
                                  dist_ow_ca_mat,
                                  dist_ow_an_mat), axis=1)

    idx_neighbors_mat = np.argsort(dist_ow_mat, axis=1)[:,1:5]

    #mask_water = (idx_neighbors_mat >= idx_ow_i) & (idx_neighbors_mat <= idx_ow_f)
    mask_ca = (idx_neighbors_mat >= idx_ca_i) & (idx_neighbors_mat <= idx_ca_f)
    mask_an = (idx_neighbors_mat >= idx_an_i) & (idx_neighbors_mat <= idx_an_f)

    neighbors_mat = np.zeros_like(idx_neighbors_mat, dtype=int)
    neighbors_mat[mask_ca] = 1
    neighbors_mat[mask_an] = 2

    for j in range(n_ow):
        status = get_status(Counter(neighbors_mat[j]))
        status_mat[idx,j] = status    

    idx += 1


np.savez(args.o, status=status_mat)

