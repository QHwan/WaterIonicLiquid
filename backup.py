import argparse
import MDAnalysis as md
import numpy as np
import numba as nb
import MDAnalysis.analysis.distances as mdanadist
from tqdm import tqdm
from collections import Counter

@nb.jit
def get_status(dist, idx_dist):
    status = 0
    for i in range(1, len(dist)):
        d = dist[i]
        idx_ow = idx_dist[i]

        if d > 5.8:
            return status

        if idx_ow >= n_ow:
            status += 1

    return(status)




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

    for j in range(n_ow):
        #idx_neighbors_mat = np.argsort(dist_ow_mat, axis=1)[:,1:5]

        dist_ow_vec = dist_ow_mat[j]
        sorted_dist_ow_vec = np.sort(dist_ow_vec)
        idx_sorted_dist_ow_vec = np.argsort(dist_ow_vec)

        
        status_mat[idx, j] = get_status(sorted_dist_ow_vec, idx_sorted_dist_ow_vec)


    idx += 1


np.savez(args.o, status=status_mat)

