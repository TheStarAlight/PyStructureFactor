# from PyStructureFactor import get_structure_factor
import numpy as np
import pyscf
n_beta  = 90
n_gamma = 1
molO2 = pyscf.M(atom="O 0,0,-1.14095; O 0,0,1.14095", unit="B", basis="pc-4", spin=2, symmetry=True)    # must add `symmetry=True` for molecules with degenerate HOMOs.
# === HOMO & HOMO-1 are degenerate, need to distinguish HOMO-xz & HOMO-yz
task = pyscf.scf.UHF(molO2).run()
mo_occ = task.mo_occ
index = get_homo_index(mo_occ[0])
coeff = task.mo_coeff[0]
coeff = numpy.expand_dims(coeff[:, index], axis=0)
yz_plane = numpy.array([[0,0.2,0.2],[0,1,0.5],[0,5,6]]) # choose some pts on x=0 plane to evaluate the wfn
ao = molO2.eval_gto('GTOval', yz_plane)
wfn = numpy.dot(ao, coeff.T)
index_yz = 0
index_xz = 0
if wfn.all() <= 1e-9:
    index_xz = -1
else:
    index_yz = -1
# =================================
beta_grid = np.linspace(0, numpy.pi, n_beta)
O2_HOMOxz_G01_conv_1 = get_structure_factor(mol = molO2, rel_homo_index = index_xz, channel = (0,1),
                           lmax = 10, hf_method = "UHF",
                           atom_grid_level = 1,
                           orient_grid_size = (n_beta, n_gamma))
print("O2_HOMOxz_G01_conv_1 finished.")
O2_HOMOxz_G01_conv_3 = get_structure_factor(mol = molO2, rel_homo_index = index_xz, channel = (0,1),
                           lmax = 10, hf_method = "UHF",
                           atom_grid_level = 3,
                           orient_grid_size = (n_beta, n_gamma))
print("O2_HOMOxz_G01_conv_3 finished.")
O2_HOMOxz_G01_conv_5 = get_structure_factor(mol = molO2, rel_homo_index = index_xz, channel = (0,1),
                           lmax = 10, hf_method = "UHF",
                           atom_grid_level = 5,
                           orient_grid_size = (n_beta, n_gamma))
print("O2_HOMOxz_G01_conv_5 finished.")
O2_HOMOxz_G01_conv_7 = get_structure_factor(mol = molO2, rel_homo_index = index_xz, channel = (0,1),
                           lmax = 10, hf_method = "UHF",
                           atom_grid_level = 7,
                           orient_grid_size = (n_beta, n_gamma))
print("O2_HOMOxz_G01_conv_7 finished.")

import matplotlib.pyplot as plt
def abs2(v):
    return np.real(v*np.conj(v))
fig = plt.figure(figsize=(5,4))
plt.plot(beta_grid*180/np.pi, abs2(O2_HOMOxz_G01_conv_1), label=r'grid_level=1')
plt.plot(beta_grid*180/np.pi, abs2(O2_HOMOxz_G01_conv_3), label=r'grid_level=3')
plt.plot(beta_grid*180/np.pi, abs2(O2_HOMOxz_G01_conv_5), label=r'grid_level=5')
plt.plot(beta_grid*180/np.pi, abs2(O2_HOMOxz_G01_conv_7), label=r'grid_level=7')
plt.xlabel(r'$\beta$ (deg)')
plt.ylabel(r'HOMO-xz $|G_{01}|^{2}$ (a.u.)')
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.title("$\mathrm{O}_2$",x=0.9,y=0.85,size=24)
plt.xlim([0, 180])
plt.ylim(bottom=0)
plt.legend(loc='lower right')
# draw subplot
left,bottom,width,height = 0.1,0.25,0.5,0.4
subax = fig.add_axes([left,bottom,width,height])
subax.plot(beta_grid*180/np.pi, abs2(O2_HOMOxz_G01_conv_1))
subax.plot(beta_grid*180/np.pi, abs2(O2_HOMOxz_G01_conv_3))
subax.plot(beta_grid*180/np.pi, abs2(O2_HOMOxz_G01_conv_5))
subax.plot(beta_grid*180/np.pi, abs2(O2_HOMOxz_G01_conv_7))
subax.set_xlim([60, 90])
subax.set_ylim([0,2])
plt.savefig("./O2_ConvTest.pdf")
plt.show()