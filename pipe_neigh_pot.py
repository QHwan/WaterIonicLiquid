import argparse
import MDAnalysis as md
import numpy as np
import MDAnalysis.analysis.distances as mdanadist
from tqdm import tqdm

from analysis.parameter import Parameter, sig_matrix, eps_matrix, q_matrix
from analysis.potential import potential_between_molecule

param = Parameter()

q_ow = param.charge_dict['OW']
q_hw = param.charge_dict['HW1']
q_na = param.charge_dict['NA']
q_cl = param.charge_dict['CL']

sig_ow = param.sigma_dict['OW']
sig_na = param.sigma_dict['NA']
sig_cl = param.sigma_dict['CL']

eps_ow = param.epsilon_dict['OW']
eps_na = param.epsilon_dict['NA']
eps_cl = param.epsilon_dict['CL']

sig_sol = np.array([sig_ow, 0, 0])
eps_sol = np.array([eps_ow, 0, 0])
q_sol = np.array([q_ow, q_hw, q_hw])

sig_sol_sol = sig_matrix(sig_sol, sig_sol)
sig_sol_na = sig_matrix(sig_sol, [sig_na])
sig_sol_cl = sig_matrix(sig_sol, [sig_cl])
eps_sol_sol = eps_matrix(eps_sol, eps_sol)
eps_sol_na = eps_matrix(eps_sol, [eps_na])
eps_sol_cl = eps_matrix(eps_sol, [eps_cl])
q_sol_sol = q_matrix(q_sol, q_sol)
q_sol_na = q_matrix(q_sol, [q_na])
q_sol_cl = q_matrix(q_sol, [q_cl])


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
sol = u.select_atoms("name OW or name HW1 or name HW2")

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


pot_mat = []

for i, ts in tqdm(enumerate(u.trajectory), total=len(u.trajectory)):
    box = ts.dimensions

    x_ow = ow.positions
    x_ca = ca.positions
    x_an = an.positions
    x_sol = sol.positions

    mdanadist.distance_array(x_ow, x_ow, box, 
                             result=dist_ow_ow_mat,
                             backend='OpenMP')
    mdanadist.distance_array(x_ow, x_ca, box, 
                             result=dist_ow_ca_mat,
                             backend='OpenMP')
    mdanadist.distance_array(x_ow, x_an, box, 
                             result=dist_ow_an_mat,
                             backend='OpenMP')

    dist_ow_mat = np.concatenate((dist_ow_ow_mat,
                                  dist_ow_ca_mat,
                                  dist_ow_an_mat), axis=1)

    idx_neighbors_mat = np.argsort(dist_ow_mat, axis=1)[:,1:20+1]

    mask_ca = (idx_neighbors_mat >= idx_ca_i) & (idx_neighbors_mat <= idx_ca_f)
    mask_an = (idx_neighbors_mat >= idx_an_i) & (idx_neighbors_mat <= idx_an_f)

    neighbors_mat = np.zeros_like(idx_neighbors_mat, dtype=int)
    neighbors_mat[mask_ca] = 1
    neighbors_mat[mask_an] = 2

    for j in range(n_ow):
        neighbors_vec = neighbors_mat[j]
        idx_neighbors_vec = idx_neighbors_mat[j]

        pot_vec = []
        pot = 0.
        for k in range(len(neighbors_vec)):

            idx_nei = idx_neighbors_vec[k]

            if neighbors_vec[k] == 0:
                p_lj, p_c = potential_between_molecule(x_sol[3*j:3*j+3], 
                                                       x_sol[3*idx_nei:3*idx_nei+3], 
                                                       sig_sol_sol,
                                                       eps_sol_sol,
                                                       q_sol_sol,
                                                       box)
                pot += (p_lj + p_c) * 0.5

            if neighbors_vec[k] == 1:
                idx_nei -= n_ow
                p_lj, p_c = potential_between_molecule(x_sol[3*j:3*j+3],
                                                       np.reshape(x_ca[idx_nei], (1, -1)),
                                                       sig_sol_na,
                                                       eps_sol_na,
                                                       q_sol_na,
                                                       box)
                pot += p_lj + p_c

            if neighbors_vec[k] == 2:
                idx_nei -= n_ow + n_ca
                p_lj, p_c = potential_between_molecule(x_sol[3*j:3*j+3],
                                                       np.reshape(x_an[idx_nei], (1, -1)),
                                                       sig_sol_cl,
                                                       eps_sol_cl,
                                                       q_sol_cl,
                                                       box)
                pot += p_lj + p_c

            pot_vec.append(pot)

        pot_mat.append(pot_vec)


np.savez(args.o, pot=np.array(pot_mat))

