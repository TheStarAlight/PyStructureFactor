# C2H2_example.py
# Usage: place this file under the same path with the PyStructureFactor.py, and execute this file.
from PyStructureFactor import get_structure_factor, get_homo_index
import numpy as np
import pyscf
n_beta  = 180
n_gamma = 1
molC2H2 = pyscf.M(atom="C 0,0,0.6013; C 0,0,-0.6013; H 0,0,1.6644; H 0,0,-1.6644", basis="pc-2", spin=0, symmetry=True)
# === HOMO & HOMO-1 are degenerate, need to be recombined to HOMO-xz & HOMO-yz
task = pyscf.scf.RHF(molC2H2).run()
mo_occ = task.mo_occ
index = get_homo_index(mo_occ)
coeff = task.mo_coeff
coeff = np.expand_dims(coeff[:, index], axis=0)
yz_plane = np.array([[0,0.2,0.2],[0,1,0.5],[0,5,6]]) # choose some pts on x=0 plane to evaluate the wfn
ao = molC2H2.eval_gto('GTOval', yz_plane)
wfn = np.dot(ao, coeff.T)
index_yz = 0
index_xz = 0
if wfn.all() <= 1e-9:
    index_xz = -1
else:
    index_yz = -1
# =================================
beta_grid = np.linspace(0, np.pi, n_beta)
C2H2_HOMOyz_G00 = get_structure_factor(mol = molC2H2, orbital_index = index_yz, channel = (0,0),
                           lmax = 10, hf_method = "RHF",
                           atom_grid_level = 3,
                           orient_grid_size = (n_beta, n_gamma))
print("C2H2_HOMOyz_G00 finished.")
C2H2_HOMOxz_G01 = get_structure_factor(mol = molC2H2, orbital_index = index_xz, channel = (0,1),
                           lmax = 10, hf_method = "RHF",
                           atom_grid_level = 3,
                           orient_grid_size = (n_beta, n_gamma))
print("C2H2_HOMOxz_G01 finished.")
C2H2_HOMOyz_G01 = get_structure_factor(mol = molC2H2, orbital_index = index_yz, channel = (0,1),
                           lmax = 10, hf_method = "RHF",
                           atom_grid_level = 3,
                           orient_grid_size = (n_beta, n_gamma))
print("C2H2_HOMOyz_G01 finished.")

import matplotlib.pyplot as plt
def abs2(v):
    return np.real(v*np.conj(v))
plt.figure(figsize=(5,4))
plt.plot(beta_grid*180/np.pi, abs2(C2H2_HOMOxz_G01), label=r'HOMO-$xz$, $\nu=(0,1)$')
plt.plot(beta_grid*180/np.pi, abs2(C2H2_HOMOyz_G00), label=r'HOMO-$yz$, $\nu=(0,0)$')
plt.plot(beta_grid*180/np.pi, abs2(C2H2_HOMOyz_G01), label=r'HOMO-$yz$, $\nu=(0,1)$')
plt.xlabel(r'$\beta$ (deg)')
plt.ylabel(r'$|G_{\nu}|^{2}$ (a.u.)')
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.title("$\mathrm{C}_2 \mathrm{H}_2$",x=0.87,y=0.85,size=24)
plt.xlim([0, 180])
plt.ylim(bottom=0)
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("./C2H2_Example.pdf")
plt.show()