# CO_example.py
# Usage: place this file under the same path with the PyStructureFactor.py, and execute this file.
from PyStructureFactor import get_structure_factor, get_homo_index
import numpy as np
import pyscf
from scipy import special
n_beta  = 180
n_gamma = 1
molCO = pyscf.M(atom="C 0,0,-0.180; O 0,0,0.950", basis="pc-4", spin=0, symmetry=True)
beta_grid  = np.linspace(0, np.pi, n_beta)
CO_HOMO_G00 = get_structure_factor(mol = molCO, orbital_index = 0, channel = (0,0),
                           lmax = 10, hf_method = "RHF",
                           atom_grid_level = 3,
                           orient_grid_size = (n_beta, n_gamma))
print("CO_HOMO_G00 finished.")

# === HOMO-1 & HOMO-2 are degenerate, need to be recombined to HOMO-1-xz & HOMO-1-yz
task = pyscf.scf.RHF(molCO).run()
mo_occ = task.mo_occ
index = get_homo_index(mo_occ)-1
coeff = task.mo_coeff
coeff = np.expand_dims(coeff[:, index], axis=0)
yz_plane = np.array([[0,0.2,0.2],[0,1,0.5],[0,5,6]]) # choose some pts on x=0 plane to evaluate the wfn
ao = molCO.eval_gto('GTOval', yz_plane)
wfn = np.dot(ao, coeff.T)
index_yz = -1
index_xz = -1
if np.all(abs(wfn) <= 1e-9):
    index_xz = -2
else:
    index_yz = -2
# =================================
CO_HOMOm1yz_G00 = get_structure_factor(mol = molCO, orbital_index = index_yz, channel = (0,0),
                           lmax = 10, hf_method = "RHF",
                           atom_grid_level = 3,
                           orient_grid_size = (n_beta, n_gamma))
print("CO_HOMOm1yz_G00 finished.")

import matplotlib.pyplot as plt
def abs2(v):
    return np.real(v*np.conj(v))
fig = plt.figure(figsize=(6.4,4.8))
plt.plot(beta_grid*180/np.pi, np.abs2(CO_HOMO_G00), label=r'HOMO')
plt.plot(beta_grid*180/np.pi, np.abs2(CO_HOMOm1yz_G00), label=r'HOMO-1 ($yz$)')
plt.xlabel(r'$\beta$ (deg)')
plt.ylabel(r'$|G_{\nu}|^{2}$ (a.u.)')
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.title("$\mathrm{H}_2$",x=0.9,y=0.85,size=30)
plt.xlim([0, 180])
plt.ylim(bottom=0,top=45)
plt.legend()
plt.tight_layout()
plt.savefig("./CO_Example.pdf")
plt.show()