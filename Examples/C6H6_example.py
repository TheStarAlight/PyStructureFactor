# C6H6_example.py
# Usage: place this file under the same path with the PyStructureFactor.py, and execute this file.
from PyStructureFactor import get_structure_factor, get_homo_index
import numpy as np
import pyscf
n_beta  = 180
n_gamma = 360
molC6H6 = pyscf.M(atom='''
                        C -0.7,  0,  1.2124
                        C  0.7,  0,  1.2124
                        C -0.7,  0, -1.2124
                        C  0.7,  0, -1.2124
                        C  1.4,  0,  0
                        C -1.4,  0,  0
                        H -1.24, 0,  1.2124
                        H  1.24, 0,  1.2124
                        H -1.24, 0, -1.2124
                        H  1.24, 0, -1.2124
                        H -2.48, 0,  0
                        H  2.48, 0,  0
                        ''',
                basis='pc-1', symmetry=True)    # must add `symmetry=True` for molecules with degenerate HOMOs.
# === HOMO & HOMO-1 are degenerate, need to be recombined to HOMO-xz & HOMO-yz
task = pyscf.scf.RHF(molC6H6).run()
mo_occ = task.mo_occ
index = get_homo_index(mo_occ)
coeff = task.mo_coeff
coeff = np.expand_dims(coeff[:, index], axis=0)
yz_plane = np.array([[0,0.2,0.2],[0,1,0.5],[0,5,6]]) # choose some pts on x=0 plane to evaluate the wfn
ao = molC6H6.eval_gto('GTOval', yz_plane)
wfn = np.dot(ao, coeff.T)
index_yz = 0
index_xz = 0
if wfn.all() <= 1e-9:
    index_xz = -1
else:
    index_yz = -1
# =================================
beta_grid  = np.linspace(0, np.pi, n_beta)
gamma_grid = np.linspace(0, 2*np.pi, n_gamma)
C6H6_HOMOxz_G00 = get_structure_factor(mol = molC6H6, rel_homo_index = index_xz, channel = (0,0),
                           lmax = 10, hf_method = "RHF",
                           atom_grid_level = 3,
                           orient_grid_size = (n_beta, n_gamma))
print("C6H6_HOMOxz_G00 finished.")
C6H6_HOMOyz_G00 = get_structure_factor(mol = molC6H6, rel_homo_index = index_yz, channel = (0,0),
                           lmax = 10, hf_method = "RHF",
                           atom_grid_level = 3,
                           orient_grid_size = (n_beta, n_gamma))
print("C6H6_HOMOyz_G00 finished.")

import matplotlib.pyplot as plt
def abs2(v):
    return np.real(v*np.conj(v))

fig=plt.figure(figsize=(8,3))
plt.xlim([0, 360])
plt.ylim([0, 180])
norm = plt.Normalize(vmin=-2, vmax=2)
heatmap = plt.imshow(np.real(C6H6_HOMOxz_G00), cmap='RdBu', interpolation='nearest', extent=(0,360,0,180), norm=norm)
fig.colorbar(heatmap,ax=fig.axes[0],label='$G_{00}$ (a.u.)')
plt.xlabel(r'$\beta$ (deg)')
plt.ylabel(r'$\gamma$ (deg)')
plt.yticks(np.arange(0,180+1,30))
plt.xticks(np.arange(0,360+1,45))
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.title("$\mathrm{C}_6 \mathrm{H}_6$",x=0.88,y=0.8,size=24)
plt.text(285,130,"HOMO-$xz$",size=12)
plt.text(180,140,"$-$",size=16,color='white',horizontalalignment='center')
plt.text(180,35,"$+$",size=16,color='white',horizontalalignment='center')
plt.text(6,140,"$+$",size=16,color='white',horizontalalignment='center')
plt.text(6,35,"$-$",size=16,color='white',horizontalalignment='center')
plt.text(354,140,"$+$",size=16,color='white',horizontalalignment='center')
plt.text(354,35,"$-$",size=16,color='white',horizontalalignment='center')
plt.savefig("./C6H6_HOMOxz_Example.pdf")
plt.show()

fig=plt.figure(figsize=(8,3))
plt.xlim([0, 360])
plt.ylim([0, 180])
norm = plt.Normalize(vmin=-2, vmax=2)
heatmap = plt.imshow(np.real(C6H6_HOMOyz_G00), cmap='RdBu', interpolation='nearest', extent=(0,360,0,180), norm=norm)
fig.colorbar(heatmap,ax=fig.axes[0],label='$G_{00}$ (a.u.)')
plt.xlabel(r'$\beta$ (deg)')
plt.ylabel(r'$\gamma$ (deg)')
plt.yticks(np.arange(0,180+1,30))
plt.xticks(np.arange(0,360+1,45))
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.title("$\mathrm{C}_6 \mathrm{H}_6$",x=0.88,y=0.8,size=24)
plt.text(285,130,"HOMO-$yz$",size=12)
plt.text(90,135,"$+$",size=16,color='white',horizontalalignment='center')
plt.text(90,40,"$-$",size=16,color='white',horizontalalignment='center')
plt.text(270,135,"$-$",size=16,color='white',horizontalalignment='center')
plt.text(270,40,"$+$",size=16,color='white',horizontalalignment='center')
plt.savefig("./C6H6_HOMOyz_Example.pdf")
plt.show()