<<<<<<< HEAD
import argparse
import numpy as np
import numba as nb
from tqdm import tqdm
=======
import numpy as np
import numba as nb
>>>>>>> 2b9ec1787887c53336aafff14a66052b412e0682
import matplotlib.pyplot as plt
import MDAnalysis as md
import MDAnalysis.analysis.distances as mdanadist

<<<<<<< HEAD
from analysis.parameter import Parameter, sig_matrix, eps_matrix, q_matrix
from analysis.potential import potential_between_molecule

import time


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






@nb.jit
def potential_frame(x_sol, x_na, x_cl, box):
    n_ow = int(len(x_sol)/3)
    n_na = len(x_na)
    n_cl = len(x_cl)

    p_owow_vec = np.zeros(n_ow)
    p_owna_vec = np.zeros(n_ow)
    p_owcl_vec = np.zeros(n_ow)
    for i in range(n_ow):
        for j in range(n_ow):
            if i == j:
                continue
            p_lj, p_c = potential_between_molecule(x_sol[3*i:3*i+3], 
                                                   x_sol[3*j:3*j+3], 
                                                   sig_sol_sol,
                                                   eps_sol_sol,
                                                   q_sol_sol,
                                                   box)
            p_owow_vec[i] += (p_lj + p_c) * 0.5

        for j in range(n_na):
            p_lj, p_c = potential_between_molecule(x_sol[3*i:3*i+3],
                                                   np.reshape(x_na[j], (1, -1)),
                                                   sig_sol_na,
                                                   eps_sol_na,
                                                   q_sol_na,
                                                   box)
            p_owna_vec[i] += p_lj + p_c

        for j in range(n_cl):
            p_lj, p_c = potential_between_molecule(x_sol[3*i:3*i+3],
                                                   np.reshape(x_cl[j], (1, -1)),
                                                   sig_sol_cl,
                                                   eps_sol_cl,
                                                   q_sol_cl,
                                                   box)
            p_owcl_vec[i] += p_lj + p_c

    return(p_owow_vec, p_owna_vec, p_owcl_vec)


parser = argparse.ArgumentParser()
parser.add_argument('--f')
parser.add_argument('--s')
parser.add_argument('--o')
args = parser.parse_args()

u = md.Universe(args.s, args.f)

ow = u.select_atoms('name OW')
sol = u.select_atoms('name OW or name HW1 or name HW2')
na = u.select_atoms('name NA')
cl = u.select_atoms('name CL')

n_ow = len(ow)

p_owow_mat = np.zeros((len(u.trajectory), n_ow))
p_owna_mat = np.zeros_like(p_owow_mat)
p_owcl_mat = np.zeros_like(p_owow_mat)

for i, ts in tqdm(enumerate(u.trajectory), total=len(u.trajectory)):
    box = ts.dimensions
    x_sol = sol.positions
    x_na = na.positions
    x_cl = cl.positions

    p_owow_mat[i], p_owna_mat[i], p_owcl_mat[i] = potential_frame(x_sol, x_na, x_cl, box)

np.savez(args.o, pot_owow=p_owow_mat,
                 pot_owna=p_owna_mat,
                 pot_owcl=p_owcl_mat)
=======
from potential import potential_sol_sol

import time

@nb.njit(fastmath=True)
def potential_sol_sol_frame(x_sol, box):
    n_ow = int(len(x_sol)/3)
    p = 0
    for i in range(n_ow):
        for j in range(i+1, n_ow):
            p_lj, p_c = potential_sol_sol(x_sol[3*i:3*i+3], x_sol[3*j:3*j+3], box)
            p += p_lj + p_c
    return(p)

u = md.Universe('trj/md_0m.tpr',
                'trj/md_0m_1000frame.xtc')

ow = u.select_atoms('name OW')
sol = u.select_atoms('name OW or name HW1 or name HW2')

n_ow = len(ow)

for i in range(5):
    start_time = time.time()

    ts = u.trajectory[i]
    box = ts.dimensions
    x_sol = sol.positions

    p = potential_sol_sol_frame(x_sol, box)

    print(p)
    print(time.time() - start_time)
>>>>>>> 2b9ec1787887c53336aafff14a66052b412e0682



