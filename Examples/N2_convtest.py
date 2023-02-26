# N2_convtest.py
# Usage: place this file under the same path with the PyStructureFactor.py, and execute this file.

from PyStructureFactor import get_structure_factor
import numpy as np
import pyscf
from scipy import special
n_beta  = 180
n_gamma = 1
molN2 = pyscf.M(atom="N 0,0,-0.5497; N 0,0,0.5497", basis="pc-4", spin=0)
beta_grid = np.linspace(0, np.pi, n_beta)
N2G00_HOMOm2_conv_1 = get_structure_factor(mol = molN2, orbital_index = -2, channel = (0,0),
                           lmax = 10, hf_method = "RHF",
                           atom_grid_level = 1,
                           orient_grid_size = (n_beta, n_gamma))
print("N2G00_HOMOm2_conv_1 finished.")
N2G00_HOMOm2_conv_3 = get_structure_factor(mol = molN2, orbital_index = -2, channel = (0,0),
                           lmax = 10, hf_method = "RHF",
                           atom_grid_level = 3,
                           orient_grid_size = (n_beta, n_gamma))
print("N2G00_HOMOm2_conv_3 finished.")
N2G00_HOMOm2_conv_5 = get_structure_factor(mol = molN2, orbital_index = -2, channel = (0,0),
                           lmax = 10, hf_method = "RHF",
                           atom_grid_level = 5,
                           orient_grid_size = (n_beta, n_gamma))
print("N2G00_HOMOm2_conv_5 finished.")
N2G00_HOMOm2_conv_7 = get_structure_factor(mol = molN2, orbital_index = -2, channel = (0,0),
                           lmax = 10, hf_method = "RHF",
                           atom_grid_level = 7,
                           orient_grid_size = (n_beta, n_gamma))
print("N2G00_HOMOm2_conv_7 finished.")

import matplotlib as mpl
import matplotlib.pyplot as plt
def abs2(v):
    return np.real(v*np.conj(v))
def N2ref(n_beta):
    m = 0
    def Theta(beta):
        T = np.sqrt((2*l+1)*special.factorial(l - m)/2/special.factorial(l + m))*special.lpmv(m, l, np.cos(beta))
        return T
    l = np.arange(0,8+1,1)
    cN2 = np.array([4.993,0.0,1.699,0.0,0.0,0.0,7.839e-4,0.0,7.773e-6])
    g_beta = []
    beta1 = np.linspace(0,np.pi,n_beta)
    for beta in beta1:
        G = Theta(beta)
        G_part = np.expand_dims(G, axis=1)   #(1,6)
        cref = np.expand_dims(cN2,axis=0)  # (6,1)
        g_beta1 = np.dot(cref,G_part)[0]
        g_beta.append(g_beta1**2)
    g_beta = np.array(g_beta)
    return g_beta

fig = plt.figure(figsize=(6.4,4.8))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
plt.plot(beta_grid*180/np.pi, abs2(N2G00_HOMOm2_conv_1), label=r'grid_level=1', color=colors[0])
plt.plot(beta_grid*180/np.pi, abs2(N2G00_HOMOm2_conv_3), label=r'grid_level=3', color=colors[1])
plt.plot(beta_grid*180/np.pi, abs2(N2G00_HOMOm2_conv_5), label=r'grid_level=5', color=colors[2])
plt.plot(beta_grid*180/np.pi, abs2(N2G00_HOMOm2_conv_7), label=r'grid_level=7', color=colors[3])
plt.plot(beta_grid*180/np.pi, N2ref(n_beta), linestyle='--', label=r'ref')
plt.xlabel(r'$\beta$ (deg)')
plt.ylabel(r'HOMO-2 $|G_{00}|^{2}$ (a.u.)')
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.title("$\mathrm{N}_2$",x=0.9,y=0.85,size=24)
plt.xlim([0, 180])
plt.ylim([0, 45])
plt.legend(loc='lower left',frameon=False)
plt.tight_layout()
# draw subplot's zoomin region
rect = mpl.patches.Rectangle((31,25),1,1,edgecolor='black',facecolor='none',zorder=2)
fig.axes[0].add_patch(rect)
line1 = mpl.lines.Line2D((32,61),(26,38),color='gray',linewidth=1)
fig.axes[0].add_line(line1)
line2 = mpl.lines.Line2D((32,61),(25,25.5),color='gray',linewidth=1)
fig.axes[0].add_line(line2)
# draw subplot
left,bottom,width,height = 0.4,0.6,0.24,0.22
subax = fig.add_axes([left,bottom,width,height])
subax.plot(beta_grid*180/np.pi, abs2(N2G00_HOMOm2_conv_1))
subax.plot(beta_grid*180/np.pi, abs2(N2G00_HOMOm2_conv_3))
subax.plot(beta_grid*180/np.pi, abs2(N2G00_HOMOm2_conv_5))
subax.plot(beta_grid*180/np.pi, abs2(N2G00_HOMOm2_conv_7))
subax.plot(beta_grid*180/np.pi, N2ref(n_beta), linestyle='--')
subax.set_xlim([31,32])
subax.set_ylim([25,26])
plt.savefig("./N2_ConvTest.pdf")
plt.show()