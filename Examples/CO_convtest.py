# CO_convtest.py
# Usage: place this file under the same path with the PyStructureFactor.py, and execute this file.
from PyStructureFactor import get_structure_factor
import numpy as np
import pyscf
from scipy import special
n_beta  = 180
n_gamma = 1
molCO = pyscf.M(atom="C 0,0,-0.180; O 0,0,0.950", basis="pc-4", spin=0)
beta_grid = np.linspace(0, np.pi, n_beta)
COG00_conv_3 = get_structure_factor(mol = molCO, orbital_index = 0, channel = (0,0),
                           lmax = 10, hf_method = "RHF",
                           atom_grid_level = 3,
                           orient_grid_size = (n_beta, n_gamma))
print("COG00_conv_3 finished.")
COG00_conv_5 = get_structure_factor(mol = molCO, orbital_index = 0, channel = (0,0),
                           lmax = 10, hf_method = "RHF",
                           atom_grid_level = 5,
                           orient_grid_size = (n_beta, n_gamma))
print("COG00_conv_5 finished.")
COG00_conv_7 = get_structure_factor(mol = molCO, orbital_index = 0, channel = (0,0),
                           lmax = 10, hf_method = "RHF",
                           atom_grid_level = 7,
                           orient_grid_size = (n_beta, n_gamma))
print("COG00_conv_7 finished.")

import matplotlib as mpl
import matplotlib.pyplot as plt
def abs2(v):
    return np.real(v*np.conj(v))
def COref(n_beta):
    m = 0
    def Theta(beta):
        T = np.sqrt((2*l+1)*special.factorial(l - m)/2/special.factorial(l + m))*special.lpmv(m, l, np.cos(beta))
        return T
    l = np.arange(0,11,1)
    cCO = np.array([3.346,-1.003,8.215e-1,-4.584e-1,1.688e-1,-4.802e-2,1.130e-2,-2.275e-3,3.993e-4,-6.197e-5,8.100e-6])
    g_beta = []
    beta1 = np.linspace(0,np.pi,n_beta)
    for beta in beta1:
        G = Theta(beta)
        G_part = np.expand_dims(G, axis=1)   #(1,6)
        cref = np.expand_dims(cCO,axis=0)  # (6,1)
        g_beta1 = np.dot(cref,G_part)[0]
        g_beta.append(g_beta1**2)
    g_beta = np.array(g_beta)
    return g_beta

fig = plt.figure(figsize=(6.4,4.8))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
plt.plot(beta_grid*180/np.pi, abs2(COG00_conv_3), label=r'grid_level=3', color=colors[1])
plt.plot(beta_grid*180/np.pi, abs2(COG00_conv_5), label=r'grid_level=5', color=colors[2])
plt.plot(beta_grid*180/np.pi, abs2(COG00_conv_7), label=r'grid_level=7', color=colors[3])
plt.plot(beta_grid*180/np.pi, COref(n_beta), linestyle='--', label=r'ref')
plt.xlabel(r'$\beta$ (deg)')
plt.ylabel(r'HOMO $|G_{00}|^{2}$ (a.u.)')
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.title("$\mathrm{CO}$",x=0.9,y=0.85,size=24)
plt.xlim([0, 180])
plt.ylim([0,45])
plt.legend(frameon=False)
plt.tight_layout()
# draw subplot's zoomin region
rect = mpl.patches.Rectangle((170,36),10,4,edgecolor='black',facecolor='none')
fig.axes[0].add_patch(rect)
line1 = mpl.lines.Line2D((170,104.5),(40,25),color='gray',linewidth=1)
fig.axes[0].add_line(line1)
line2 = mpl.lines.Line2D((170,104.5),(36,8.5),color='gray',linewidth=1)
fig.axes[0].add_line(line2)
# draw subplot
left,bottom,width,height = 0.25,0.3,0.35,0.3
subax = fig.add_axes([left,bottom,width,height])
subax.plot(beta_grid*180/np.pi, abs2(COG00_conv_3))
subax.plot(beta_grid*180/np.pi, abs2(COG00_conv_5))
subax.plot(beta_grid*180/np.pi, abs2(COG00_conv_7))
subax.plot(beta_grid*180/np.pi, COref(n_beta), linestyle='--')
subax.set_xlim([170,180])
subax.set_ylim([36,40])
plt.savefig("./CO_ConvTest.pdf")
plt.show()
