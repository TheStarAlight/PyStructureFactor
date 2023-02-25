# H2_minimal.py
# Usage: place this file under the same path with the PyStructureFactor.py, and execute this file.
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