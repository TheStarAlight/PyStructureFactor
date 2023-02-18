# H2_example.py
# Usage: place this file under the same path with the PyStructureFactor.py, and execute this file.
from PyStructureFactor import get_structure_factor
import numpy as np
import pyscf
n_beta  = 90
n_gamma = 1
molH2 = pyscf.M(atom="H 0,0,0.37; H 0,0,-0.37", basis="pc-4", spin=0)
beta_grid = np.linspace(0, np.pi, n_beta)
H2G00 = get_structure_factor(mol = molH2, rel_homo_index = 0, channel = (0,0),
                             lmax = 10, hf_method = "RHF",
                             atom_grid_level = 3,
                             orient_grid_size = (n_beta, n_gamma))
print("H2G00 finished.")
H2G01 = get_structure_factor(mol = molH2, rel_homo_index = 0, channel = (0,1),
                             lmax = 10, hf_method = "RHF",
                             atom_grid_level = 3,
                             orient_grid_size = (n_beta, n_gamma))
print("H2G01 finished.")
H2G10 = get_structure_factor(mol = molH2, rel_homo_index = 0, channel = (1,0),
                             lmax = 10, hf_method = "RHF",
                             atom_grid_level = 3,
                             orient_grid_size = (n_beta, n_gamma))
print("H2G10 finished.")

import matplotlib.pyplot as plt
plt.figure(figsize=(5,4))
plt.plot(beta_grid*180/np.pi, np.real(H2G00 * np.conj(H2G00)), label=r'$\nu=(0,0)$')
plt.plot(beta_grid*180/np.pi, np.real(H2G01 * np.conj(H2G01))*10, label=r'$\nu=(0,1)$ [×10]')
plt.plot(beta_grid*180/np.pi, np.real(H2G10 * np.conj(H2G10)), label=r'$\nu=(1,0)$')
plt.xlabel(r'$\beta$ (deg)')
plt.ylabel(r'$|G_{\nu}|^{2}$ (a.u.)')
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.title("$\mathrm{H}_2$",x=0.9,y=0.85,size=24)
plt.xlim([0, 180])
plt.ylim(bottom=0)
plt.legend()
plt.savefig("./H2_Example.pdf")
plt.show()