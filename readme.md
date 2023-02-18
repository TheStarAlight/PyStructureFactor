# PyStructureFactor

PyStructureFactor is a python code which implements the weak-field asymptotic theory (WFAT) to calculate the WFAT structure factor of molecules, which is crucial in calculating the tunneling ionization rate of molecules in strong laser fields.


## Installation

This program is currently distributed as a portable code. Simply place it under the same path with your program and add
```
from PyStructureFactor import get_structure_factor
import pyscf
```
to the header of your program.


## Usage

The main function of the program lies in the `get_structure_factor` method:
```
def get_structure_factor(mol,
                         rel_homo_index  = 0,
                         channel         = (0,0),
                         lmax            = 10,
                         hf_method       = 'RHF',
                         atom_grid_level = 3,
                         orient_grid_size= (90,1),
                         move_dip_zero   = True,
                         rmax            = 40)
```
### Parameters

`mol` : The PySCF molecule object. Initialized by invoking `pyscf.M` or `pyscf.gto.M`.\
`rel_homo_index` : Index of the ionizing orbital relative to the HOMO. **Default is `0`.** e.g., HOMO -> 0, LUMO -> +1, HOMO-1 -> -1, ...\
`channel` : Parabolic channel $ν=(n_ξ, m)$. **Default is `(0,0)`.**\
`lmax` : The maximum angular quantum number (larger l would be cut off) used in the sum. **Default is `10`.**\
`hf_method` : Indicates whether 'RHF' or 'UHF' should be used in molecular HF calculation. **Default is `'RHF'`.** *[!] Note: Must use 'UHF' for multiplet molecules.*\
`atom_grid_level` : Level of fineness of the grid used in integration (see also `pyscf.dft.Grid`), which controls the number of radial and angular grids, ranging from 0 to 9. **Default is `3`.**\
`orient_grid_size` : Indicates the size of $(β,γ)$ grid (in the output) in $β$,$γ$ directions respectively. **Default is `(90,1)`**. The grid is uniform, with $β$ ranging from $[0,π)$ and $γ$ ranging from $[0,2π)$.\
`move_dip_zero` : Indicates whether to move the molecule so that the dipole of the parent ion equals zero. **Default `True`.**\
`rmax` : [Keep default] Indicates the cut off limit of the radial grid points, points of radius>`rmax` would not be accounted in calculation. **Default is `40`.**

### Returns
A numpy array containing the structure factors $G_{n_ξ, m}$ of the given channel on the $(β,γ)$ orientation grid. Shape = `orient_grid_size`.

### Tips on parameter choice
1. Always specify the `basis` parameter when initialzing the PySCF molecule object and choose a basis set of high-accuracy. The calculation's reliability is sensitive to the wave function, a better basis set is crucial in obtaining a satisfying result.
2. When initializing the molecule object, for singlet molecules, specify `atom` and `basis`. If your molecule is a multiplet, you should also specify `spin=<2S>`, and set `hf_method = 'UHF'`.
3. If the ionizing orbit of your molecule has degeneracy, set `symmetry=True` in the molecule's constructor method, and the PySCF would help in splitting the degenerate orbits according to the symmetry. To identify their symmetry, see the example of O2.

## Examples

To run the examples, create a new python script file under the same path with the main program file, paste the following code below, and run the script.

### A minimal example
```
from PyStructureFactor import get_structure_factor
import numpy as np
import pyscf
n_beta  = 90
n_gamma = 1
molH2 = pyscf.M(atom="H 0,0,0.37; H 0,0,-0.37", basis="pc-1", spin=0)
beta_grid = np.linspace(0, numpy.pi, n_beta)
G_grid = get_structure_factor(mol = molH2, rel_homo_index = 0, channel = (0,0),
                              lmax = 10, hf_method = "RHF",
                              atom_grid_level = 3,
                              orient_grid_size = (n_beta, n_gamma))
```

### H2 example
```
from PyStructureFactor import get_structure_factor
import numpy as np
import pyscf
n_beta  = 90
n_gamma = 1
molH2 = pyscf.M(atom="H 0,0,0.37; H 0,0,-0.37", basis="pc-4", spin=0)
beta_grid = np.linspace(0, numpy.pi, n_beta)
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
```

### O2 example

```
from PyStructureFactor import get_structure_factor
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
O2_HOMOxz_G00 = get_structure_factor(mol = molO2, rel_homo_index = index_xz, channel = (0,0),
                           lmax = 10, hf_method = "UHF",
                           atom_grid_level = 7,
                           orient_grid_size = (n_beta, n_gamma))
print("O2_HOMOxz_G00 finished.")
O2_HOMOxz_G01 = get_structure_factor(mol = molO2, rel_homo_index = index_xz, channel = (0,1),
                           lmax = 10, hf_method = "UHF",
                           atom_grid_level = 7,
                           orient_grid_size = (n_beta, n_gamma))
print("O2_HOMOxz_G01 finished.")
O2_HOMOyz_G00 = get_structure_factor(mol = molO2, rel_homo_index = index_yz, channel = (0,0),
                           lmax = 10, hf_method = "UHF",
                           atom_grid_level = 7,
                           orient_grid_size = (n_beta, n_gamma))
print("O2_HOMOyz_G00 finished.")
O2_HOMOyz_G01 = get_structure_factor(mol = molO2, rel_homo_index = index_yz, channel = (0,1),
                           lmax = 10, hf_method = "UHF",
                           atom_grid_level = 7,
                           orient_grid_size = (n_beta, n_gamma))
print("O2_HOMOyz_G01 finished.")

import matplotlib.pyplot as plt
def abs2(v):
    return np.real(v*np.conj(v))
plt.figure(figsize=(5,4))
plt.plot(beta_grid*180/np.pi, abs2(O2_HOMOxz_G01), label=r'HOMO-xz, $\nu=(0,1)$')
plt.plot(beta_grid*180/np.pi, abs2(O2_HOMOyz_G00), label=r'HOMO-yz, $\nu=(0,0)$')
plt.plot(beta_grid*180/np.pi, abs2(O2_HOMOyz_G01), label=r'HOMO-yz, $\nu=(0,1)$')
plt.xlabel(r'$\beta$ (deg)')
plt.ylabel(r'$|G_{\nu}|^{2}$ (a.u.)')
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.title("$\mathrm{O}_2$",x=0.9,y=0.85,size=24)
plt.xlim([0, 180])
plt.ylim(bottom=0)
plt.legend()
plt.savefig("./O2_Example.pdf")
plt.show()
```

For more examples, please refer to the "./Examples/" directory.