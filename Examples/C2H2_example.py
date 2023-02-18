# C2H2_example.py
# Usage: place this file under the same path with the PyStructureFactor.py, and execute this file.
from PyStructureFactor import get_structure_factor, get_homo_index
import numpy as np
import pyscf
n_beta  = 180
n_gamma = 1
molCO = pyscf.M(atom="C 0,0,-0.180; O 0,0,0.950", basis="pc-4", spin=0, symmetry=True)
beta_grid = np.linspace(0, np.pi, n_beta)
COG00 = get_structure_factor(mol = molCO, rel_homo_index = 0, channel = (0,0),
                           lmax = 10, hf_method = "RHF",
                           atom_grid_level = 3,
                           orient_grid_size = (n_beta, n_gamma))
print("COG00 finished.")
# === HOMO-1 & HOMO-2 are degenerate, need to distinguish HOMO-1-xz & HOMO-1-yz.
task = pyscf.scf.RHF(molCO).run()
mo_occ = task.mo_occ
index = get_homo_index(mo_occ)
coeff = task.mo_coeff
coeff = np.expand_dims(coeff[:, index], axis=0)
yz_plane = np.array([[0,0.2,0.2],[0,1,0.5],[0,5,6]]) # choose some pts on x=0 plane to evaluate the wfn
ao = molCO.eval_gto('GTOval', yz_plane)
wfn = np.dot(ao, coeff.T)
index_yz = -1
index_xz = -1
if wfn.all() <= 1e-9:
    index_xz = -2
else:
    index_yz = -2
# =================================
CO_HOMOm1yz_G00 = get_structure_factor(mol = molCO, rel_homo_index = index_yz, channel = (0,0),
                           lmax = 10, hf_method = "RHF",
                           atom_grid_level = 3,
                           orient_grid_size = (n_beta, n_gamma))
print("CO_HOMOm1yz_G00 finished.")

import matplotlib.pyplot as plt
def abs2(v):
    return np.real(v*np.conj(v))
plt.figure(figsize=(5,4))
plt.plot(beta_grid*180/np.pi, abs2(COG00), label=r'HOMO')
plt.plot(beta_grid*180/np.pi, abs2(CO_HOMOm1yz_G00), label=r'HOMO-1 ($yz$)')
plt.xlabel(r'$\beta$ (deg)')
plt.ylabel(r'$|G_{00}|^{2}$ (a.u.)')
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.title("$\mathrm{CO}$",x=0.9,y=0.85,size=24)
plt.xlim([0, 180])
plt.ylim(bottom=0)
plt.legend(frameon=False)
plt.savefig("./CO_Example.pdf")
plt.show()