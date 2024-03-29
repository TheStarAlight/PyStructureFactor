# PyStructureFactor

PyStructureFactor is a python code which implements the weak-field asymptotic theory (WFAT) to calculate the WFAT structure factor of molecules, which is crucial in calculating the tunneling ionization rate of molecules in strong laser fields.

Our article which introduces the WFAT theory and the program is now available on [*Comput. Phys. Commun.*, 108882 (2023)](https://www.sciencedirect.com/science/article/pii/S0010465523002278). DOI: [10.1016/j.cpc.2023.108882](https://doi.org/10.1016/j.cpc.2023.108882)

## Installation

This program is currently distributed as a portable code. Simply place it under the same path with your program and add
```py
from PyStructureFactor import get_structure_factor
import pyscf
```
to the header of your program.

### Dependencies
The program depends on the following python packages:
`numpy`, `scipy`, `pyscf` and `wigner`.

To run the examples, `matplotlib` is also required for plotting.

### Troubleshooting

- *Unable to install the `wigner` package*: Try a different package source or directly install from a whl file (provided in the `./misc/` directory).
- *`Numpy` broken support for the dtype*: Try switching to another `numpy` version. [`numpy` v1.24.1 runs well on the WSL Ubuntu 22.04 LTS (@`pyscf` v2.3.0), while under `numpy` v1.25.0 this problem occurs]

## Usage

The main function of the program lies in the `get_structure_factor` method:
```py
def get_structure_factor(mol,
                         orbital_index   = 0,
                         channel         = (0,0),
                         lmax            = 10,
                         hf_method       = 'RHF',
                         casscf_conf     = None,
                         atom_grid_level = 3,
                         orient_grid_size= (90,1),
                         move_dip_zero   = True,
                         rmax            = 40)
```
### Parameters

- `mol` : The PySCF molecule object. Initialized by invoking `pyscf.M` or `pyscf.gto.M`.

- `orbital_index` : Index of the ionizing orbital relative to the HOMO. **Default is `0`.** e.g., HOMO -> 0, LUMO -> +1, HOMO-1 -> -1, ...

- `channel` : Parabolic channel $ν=(n_ξ, m)$. **Default is `(0,0)`.**

- `lmax` : The cut-off limit of the angular quantum number (larger l would be cut off) used in the summation. **Default is `10`.**

- `hf_method` : Indicates whether 'RHF' or 'UHF' should be used in molecular HF calculation. **Default is `'RHF'`.** *[!] Note: Must use 'UHF' for open-shell molecules.*

- `casscf_conf` : Configuration of CASSCF calculation consisting of (n_active_orb, n_active_elec). Specifying `None` (by default) indicates employing HF instead of CASSCF.

- `atom_grid_level` : Level of fineness of the grid used in integration (see also `pyscf.dft.Grid`), which controls the number of radial and angular grids around each atom in the evaluation of the integration, ranging from 0 to 9. **Default is `3`.**

- `orient_grid_size` : Indicates the size of the output $(\beta,\gamma)$ grid which defines the orientation of the molecule with respect to the polarization direction of the laser field. **Default is `(90,1)`**. The grid is uniform, with $\beta$ ranging from $[0,\pi)$ and $γ$ ranging from $[0,2\pi)$. Setting the $\gamma$ grid count to 1 indicates that $\gamma$ would be zero throughout the calculation.

- `move_dip_zero` : Indicates whether to move the molecule so that the dipole of the parent ion vanishes. **Default `True`.**

- `rmax` : [Keep default] Indicates the cut off limit of the radial grid points, points of radius>`rmax` would not be accounted in calculation. **Default is `40`.**

### Returns
A numpy array containing the structure factors $G_{n_ξ,\ m}$ of the given channel on the $(\beta,\gamma)$ orientation grid. Shape = `orient_grid_size`.

### Tips on parameter choice
1. Always specify the `basis` parameter when initialzing the PySCF molecule object and choose a basis set of high-accuracy. The calculation's reliability is sensitive to the wave function, a better basis set is crucial in obtaining a satisfying result.
2. When initializing the molecule object, for close-shell molecules, specify `atom` and `basis`. If your molecule is an open-shell system, you should also specify `spin=<2S>`, and set `hf_method='UHF'`.

## Examples

To run the examples, create a new python script file under the same path with the main program file, paste the following code below, and run the script.

### A minimal example
```py
from PyStructureFactor import get_structure_factor
import numpy as np
import pyscf
n_beta  = 90
n_gamma = 1
molH2 = pyscf.M(atom="H 0,0,0.37; H 0,0,-0.37", basis="pc-1", spin=0)
beta_grid = np.linspace(0, np.pi, n_beta)
G_grid = get_structure_factor(mol = molH2, orbital_index = 0, channel = (0,0),
                              lmax = 10, hf_method = "RHF",
                              atom_grid_level = 3,
                              orient_grid_size = (n_beta, n_gamma))
```
This example took 1.2 sec. to finish on an *AMD Ryzen 9 7950X CPU* on *WSL Ubuntu 22.04 LTS*.

### H2 example

Note: this example also requires the `matplotlib` package.

```py
from PyStructureFactor import get_structure_factor
import numpy as np
import pyscf
n_beta  = 90
n_gamma = 1
molH2 = pyscf.M(atom="H 0,0,0.37; H 0,0,-0.37", basis="pc-4", spin=0)
beta_grid = np.linspace(0, np.pi, n_beta)
H2G00 = get_structure_factor(mol = molH2, orbital_index = 0, channel = (0,0),
                             lmax = 10, hf_method = "RHF",
                             atom_grid_level = 3,
                             orient_grid_size = (n_beta, n_gamma))
print("H2G00 finished.")
H2G01 = get_structure_factor(mol = molH2, orbital_index = 0, channel = (0,1),
                             lmax = 10, hf_method = "RHF",
                             atom_grid_level = 3,
                             orient_grid_size = (n_beta, n_gamma))
print("H2G01 finished.")
H2G10 = get_structure_factor(mol = molH2, orbital_index = 0, channel = (1,0),
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
plt.tight_layout()
plt.savefig("./H2_Example.pdf")
plt.show()
```
This example took 16 sec. to finish on an *AMD Ryzen 9 7950X CPU* on *WSL Ubuntu 22.04 LTS*.

### O2 example

Note: this example also requires the `matplotlib` package.
Remember that for molecules with non-zero total spin $S$, specify the spin in the `pyscf.M` with `spin=<2S>`.

```py
from PyStructureFactor import get_structure_factor, get_homo_index
import numpy as np
import pyscf
n_beta  = 90
n_gamma = 1
molO2 = pyscf.M(atom="O 0,0,-1.14095; O 0,0,1.14095", unit="B", basis="pc-4", spin=2, symmetry=True)
# === HOMO & HOMO-1 are degenerate, need to distinguish HOMO-xz & HOMO-yz
task = pyscf.scf.UHF(molO2).run()
mo_occ = task.mo_occ
index = get_homo_index(mo_occ[0])
coeff = task.mo_coeff[0]
coeff = np.expand_dims(coeff[:, index], axis=0)
yz_plane = np.array([[0,0.2,0.2],[0,1,0.5],[0,5,6]]) # choose some pts on x=0 plane to evaluate the wfn
ao = molO2.eval_gto('GTOval', yz_plane)
wfn = np.dot(ao, coeff.T)
index_yz = 0
index_xz = 0
if np.all(abs(wfn) <= 1e-9):
    index_xz = -1
else:
    index_yz = -1
# =================================
beta_grid = np.linspace(0, np.pi, n_beta)
O2_HOMOxz_G01 = get_structure_factor(mol = molO2, orbital_index = index_xz, channel = (0,1),
                           lmax = 10, hf_method = "UHF",
                           atom_grid_level = 7,
                           orient_grid_size = (n_beta, n_gamma))
print("O2_HOMOxz_G01 finished.")
O2_HOMOyz_G00 = get_structure_factor(mol = molO2, orbital_index = index_yz, channel = (0,0),
                           lmax = 10, hf_method = "UHF",
                           atom_grid_level = 7,
                           orient_grid_size = (n_beta, n_gamma))
print("O2_HOMOyz_G00 finished.")
O2_HOMOyz_G01 = get_structure_factor(mol = molO2, orbital_index = index_yz, channel = (0,1),
                           lmax = 10, hf_method = "UHF",
                           atom_grid_level = 7,
                           orient_grid_size = (n_beta, n_gamma))
print("O2_HOMOyz_G01 finished.")

import matplotlib.pyplot as plt
def abs2(v):
    return np.real(v*np.conj(v))
plt.figure(figsize=(5,4))
plt.plot(beta_grid*180/np.pi, abs2(O2_HOMOxz_G01), label=r'HOMO-$xz$, $\nu=(0,1)$')
plt.plot(beta_grid*180/np.pi, abs2(O2_HOMOyz_G00), label=r'HOMO-$yz$, $\nu=(0,0)$')
plt.plot(beta_grid*180/np.pi, abs2(O2_HOMOyz_G01), label=r'HOMO-$yz$, $\nu=(0,1)$')
plt.xlabel(r'$\beta$ (deg)')
plt.ylabel(r'$|G_{\nu}|^{2}$ (a.u.)')
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.title("$\mathrm{O}_2$",x=0.9,y=0.85,size=24)
plt.xlim([0, 180])
plt.ylim(bottom=0)
plt.legend()
plt.tight_layout()
plt.savefig("./O2_Example.pdf")
plt.show()
```
This example took about 6.5 min. to finish on an *AMD Ryzen 9 7950X CPU* on *WSL Ubuntu 22.04 LTS*.

For more examples, please refer to the `./Examples/` directory.
