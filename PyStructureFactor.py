import numpy
import math
import multiprocessing as mp
from multiprocessing.dummy import Pool as ThreadPool
from pyscf import df, scf
from scipy import special
from sympy import symbols
from sympy.physics.wigner import wigner_d_small
from pyscf import gto, lib, dft
from pyscf.dft import radi


def convert_sph2cart(xyz):
    """To transform Spherical coordinates to Cartesian coordinates

     Parameters:
         xyz: 2D float array, shape (N,3)
         Cartesian coordinates [x,y,z]

     Returns:
         2D float array, shape (N,3)
         Spherical coordinates [r,theta,phi] corresponding to [x,y,z]

    """
    ptsnew = numpy.zeros(xyz.shape)
    xy2 = xyz[:, 0] ** 2 + xyz[:, 1] ** 2
    ptsnew[:, 0] = numpy.sqrt(xy2 + xyz[:, 2] ** 2)
    ptsnew[:, 1] = numpy.arctan2(numpy.sqrt(xy2), xyz[:, 2])
    ptsnew[:, 2] = numpy.arctan2(xyz[:, 1], xyz[:, 0])
    return ptsnew


def get_homo_index(np1):
    """To find the index of HOMO orbit.
    reindex(i) is the first orbit whose occupancy number is zero, index = i - 1

    Parameters:
        np1: Orbital occupancy number, 2D array

    Returns:
        int type
        the index of HOMO orbit
    """
    for i in range(np1.size):
        if np1[i] == 0:
            return i - 1


def vc_orbit(mol, z, coeff, coords, dm, method='RHF'):
    """To calculate the core potential, Ionization orbit

    Parameters:
        mol: Mole
            Same to :func:`pyscf.Mole.build`
            Molecule to calculate the electron density for.
        z: int
            Ionization charge
        coeff: 2D array,shape(:,index)
            Molecular ionization orbit coefficient
        coords: 2D array, shape (N,3)
            Cartesian grid points(x,y,z)
        dm: ndarray
            Density matrix of molecule.
        method: str
            'RHF' or 'UHF'.Default is 'RHF'

    Returns:
        float  array_like
        the core potential.

    """
    ngrids = coords.shape[0]
    vnuc_r = z / numpy.einsum('xi,xi->x', coords, coords) ** .5
    vnuc = 0
    for i in range(mol.natm):
        r = mol.atom_coord(i)
        z = mol.atom_charge(i)
        rp = r - coords
        vnuc += z / numpy.einsum('xi,xi->x', rp, rp) ** .5
    if method == 'UHF':
        dm_vex = dm[0]
        dm_vele = dm[1] + dm[0]
    elif method == 'RHF':
        dm_vex = dm / 2
        dm_vele = dm
    else:
        raise Exception("error: method must be 'RHF' or 'UHF'")

    blksize = min(8000, ngrids)
    ao = numpy.zeros((ngrids, mol.nao))
    for ip0, ip1 in lib.prange(0, ngrids, blksize):
        ao[ip0:ip1, :] = mol.eval_gto('GTOval', coords[ip0:ip1])
    ion_orbit = numpy.dot(ao, coeff.T)
    vex_ionorbit = numpy.zeros(shape=ngrids)
    mep = numpy.zeros(shape=ngrids)
    lsrange = list(lib.prange(0, ngrids, 600))

    def part(p):
        p0 = p[0]
        p1 = p[1]
        fakemol = gto.fakemol_for_charges(coords[p0:p1])
        ints = df.incore.aux_e2(mol, fakemol)
        del fakemol
        fg = numpy.dot(coeff, ints)
        fg = fg.reshape(mol.nao, p1 - p0)
        mg = numpy.dot(ao[p0:p1], dm_vex)
        vex_ionorbit[p0:p1] = -numpy.einsum('br,br->r', fg, mg.T)
        vele = numpy.einsum('ijp,ij->p', ints, dm_vele)
        mep[p0:p1] = -vnuc[p0:p1] + vele
        del ints
        return 0

    num_cores = int(mp.cpu_count()) - 2
    pool = ThreadPool(num_cores)
    pool.map(part, lsrange)
    pool.close()
    pool.join()

    vex_ionorbit = numpy.expand_dims(vex_ionorbit, axis=1)
    mep_vnuc = numpy.expand_dims((mep + vnuc_r), axis=1)
    vc_orbit = vex_ionorbit + mep_vnuc * ion_orbit
    return vc_orbit


def omega_wvl(m, n_ep, kappa, z, l):
    """To calculate the normalization coefficient of Omega

    Parameters
        m: int
            Magnetic quantum number; must have ``l >= m >= -l``
        l: int
            Angular quantum number; must have ``l >= 0``
        n_ep: int
              Parabolic quantum numbers; must have ``n_ep >= 0``
        z: int
            Ionization charge
        kappa: float
            kappa = sqrt(-2*E0). E0 is orbital energy.
    Returns:
        float
        return the normalization coefficient of Omega
    """
    part1 = ((-1) ** (l + (abs(m) - m) / 2 + 1)) * (2 ** (l + 3 / 2)) * (
            kappa ** (z / kappa - (abs(m) + 1) / 2 - n_ep))
    part2 = math.sqrt(
        (2 * l + 1) * math.factorial(l + m) * math.factorial(l - m) * math.factorial(
            abs(m) + n_ep) * math.factorial(
            n_ep)) * math.factorial(l) / math.factorial(2 * l + 1)
    range_len = min(n_ep, l - abs(m))
    part3 = 0

    for k in range(range_len + 1):
        part3 += special.gamma(l + 1 - z / kappa + n_ep - k) / (
                math.factorial(k) * math.factorial(l - k) * math.factorial(abs(m) + k) * math.factorial(
            l - abs(m) - k) * math.factorial(n_ep - k))
    omega_norm = part1 * part2 * part3
    return omega_norm


def omega_r(kappa, z, l, coords):
    """Main part of Omega

    Parameters:
        l : int
            Angular quantum number; must have ``l >= 0``
        z : int
            Ionization charge
        kappa : float
            kappa = sqrt(-2*E0). E0 is orbital energy.

    Returns:
        float  array_like, shape(2l+1,ngrids)
    """
    coords_sph = convert_sph2cart(coords)
    m1 = [[i] for i in range(-l, l + 1)]
    r = coords_sph[:, 0]
    theta = coords_sph[:, 1]
    phi = coords_sph[:, 2]
    ylm = (special.sph_harm(m1, l, phi, theta)).conjugate()
    hypf = special.hyp1f1(l + 1 - z / kappa, 2 * l + 2, 2 * kappa * r)
    rvl = hypf * (kappa * r) ** l * numpy.exp(-kappa * r)
    rst = rvl * ylm
    omega = numpy.array(rst)
    return omega

def orbital_dip(coeff,mol):
    with mol.with_common_orig((0, 0, 0)):
        ao_dip1 = mol.intor_symmetric('int1e_r', comp=3)
    dm_01 = coeff.T * coeff
    uz1 = -numpy.einsum('xij,ji->x', ao_dip1, dm_01).real
    uz = uz1
    return uz

def get_structure_factor(mol, z=1, lmax=10,
                     update_index=0,
                     rmax = 40,
                     method='RHF',
                     atom_grid=(10,38),
                     channel=None,
                     yz_Eulerang_grid=None):
    """To caculation  the molecule structure factor.

    Parameters:
        mol : Mole
            Molecule to calculate the structure factor for.
        z : int
            Ionization charge
        lmax : int
            The maximum azimuthal quantum number
        method : str
            'RHF' or 'UHF'.Default is 'RHF'
        atom_grid : dict
            set same (radial,angular) grids for every atoms.
            Eg, atom_grid = (20,110) will generate 20 radial
            grids and 110 angular grids for every atom of the molecule.
            Set (radial, angular) grids for particular atoms.
            Eg, atom_grid = {'H': (20,110)} will generate 20 radial
            grids and 110 angular grids for H atom.
        channel : dict
            ionization channel ν=(nξ, m).Default is channel={'m': 0, 'n_ep': 0}.
            m is the magnetic quantum numbers and nξ is the parabolic quantum numbers.
        yz_Eulerang_grid : dict
            Set (β,γ)) grids for the caculation of the molecule stracture factor.
            The orientation of the molecule with respect to the field is specified by the three Euler angles (α,β,γ) defining a passive rotation
            Rˆ from the LF to the MF.
            In this calculation, we set α = 0. Default is yz_Eulerang_grid={"beta_num": 20, "gamma_num": 1}.
        update_index : int
            Ionization index = index of HOMO  +  update_index.Defualt is zero.
            Eg, LUMO oribital,update_index=1

    Returns:
        array_like

    """

    channel = {'m': 0, 'n_ep': 0} if channel is None else channel
    yz_Eulerang_grid = {"beta_num": 20, "gamma_num": 1} if yz_Eulerang_grid is None else yz_Eulerang_grid
    m = channel.get('m')
    n_ep = channel.get('n_ep')
    nbeta = yz_Eulerang_grid.get('beta_num')
    gamma_num = yz_Eulerang_grid.get('gamma_num')


    if method == 'RHF':
        mf = scf.RHF(mol).run()
        mo_occ = mf.mo_occ
        orbit_energy = mf.mo_energy
        coeff = mf.mo_coeff
    elif method == 'UHF':
        mf = scf.UHF(mol).run()
        mo_occ = mf.mo_occ[0]
        orbit_energy = mf.mo_energy[0]
        coeff = mf.mo_coeff[0]
    else:
        raise Exception("error: method must be 'RHF' or 'UHF'")

    index = get_homo_index(mo_occ) + update_index
    energy_index = orbit_energy[index]
    coeff = numpy.expand_dims(coeff[:, index], axis=0)
    kappa = math.sqrt(-2 * energy_index)
    dm = mf.make_rdm1()

    import copy
    atom = copy.deepcopy(mol._atom)
    tDict = {}
    for k in gto.Mole.build.__code__.co_varnames:
        if k in mol.__dict__ and not k.startswith("_"):
            tDict[k] = mol.__dict__[k]
    tDict["atom"] = atom
    tDict["unit"] = 'Bohr'
    mol = gto.M(**tDict)
    with mol.with_common_orig((0, 0, 0)):
        ao_dip1 = mol.intor_symmetric('int1e_r', comp=3)
    dm_01 = coeff.T * coeff
    u = -numpy.einsum('xij,ji->x', ao_dip1, dm_01).real
    dip_moment = mf.dip_moment(unit="A.U.")
    D = (u - dip_moment)
    move_distance = D/z
    atom = copy.deepcopy(mol._atom)
    for i in range(len(atom)):
        for j in range(len(atom[i][1])):
            atom[i][1][j] += move_distance[j]
    tDict = {}
    for k in gto.Mole.build.__code__.co_varnames:
        if k in mol.__dict__ and not k.startswith("_"):
            tDict[k] = mol.__dict__[k]
    tDict["atom"] = atom
    tDict["unit"] = 'Bohr'
    mol = gto.M(**tDict)
    uz1 = orbital_dip(coeff, mol)[2]

    g = dft.Grids(mol)
    g.atom_grid = atom_grid
    g.radii_adjust = radi.becke_atomic_radii_adjust
    g.atomic_radii = radi.COVALENT_RADII
    g.radi_method = radi.gauss_chebyshev
    g.prune = None
    g.build()


    weights = g.weights
    coords_ori = g.coords
    weights_ori = numpy.expand_dims(weights, axis=1)
    weights_coords_ori = numpy.hstack((weights_ori, coords_ori))
    weights_coords = weights_coords_ori[(weights_coords_ori[:,1])**2 + (weights_coords_ori[:,2])**2 + (weights_coords_ori[:,3])**2 < rmax**2]
    weights = weights_coords[:,0]
    coords = weights_coords[:,1:4]

    vc_ionorbit = vc_orbit(mol, z, coeff, coords, dm, method=method)

    def factor_part(beta_num):
        g_gamma = numpy.zeros(shape=(1, gamma_num))
        beta2 = symbols("beta2", real=True)

        def gamma_planemol(l):
            gamma = list(numpy.linspace(0, 2 * math.pi, gamma_num))
            gamma = numpy.expand_dims(gamma, axis=0)
            m1 = [[i] for i in range(-l, l + 1)]
            e_gamma = numpy.exp(-1j * numpy.dot(m1, gamma))
            return e_gamma
        for l in range(abs(m), lmax + 1):
            wvl = omega_wvl(m, n_ep, kappa, z, l)
            omega = omega_r(kappa, z, l, coords) * wvl * weights
            l_lm1_v_appendm = numpy.dot(omega, vc_ionorbit)
            gamma_np = 1 if gamma_num == 1 else gamma_planemol(l)
            l_lm1_v_mgamma = numpy.multiply(l_lm1_v_appendm, gamma_np)
            beta_wigner = wigner_d_small(l, beta2)[::-1, l - m]
            g_gamma_sub = (beta_wigner.T * l_lm1_v_mgamma)
            g_gamma = g_gamma + g_gamma_sub
        def factorpart(beta):
            g_beta_gamma = g_gamma.subs({beta2: beta})
            g_beta_gamma = numpy.array(g_beta_gamma)
            g_beta_gamma = g_beta_gamma.astype(complex)
            return g_beta_gamma
        beta = numpy.linspace(0, math.pi, beta_num)
        uz = uz1 * numpy.cos(beta)
        uz = numpy.expand_dims(uz, axis=1)
        pool1 = ThreadPool(20)
        mn = pool1.map(factorpart, beta)

        g_beta = numpy.array(mn)
        g_beta = g_beta.reshape(beta_num, gamma_num)
        structurefactor = g_beta * numpy.exp(-kappa * uz)
        pool1.close()
        pool1.join()
        return structurefactor
    return factor_part(nbeta)


if __name__ == '__main__':
    O2 = gto.M(atom='''O  0, 0, -0.403764736
                O 0, 0, 0.803764736
                ''', unit='ANG', basis='pc-1', symmetry=True, spin=2)
    nbeta = 40
    factor_xz01 = get_structure_factor(mol=O2, method='UHF',
                              channel={'n_ep': 0, 'm': 1},
                              yz_Eulerang_grid = {"beta_num": nbeta, "gamma_num": 1})

    '''plots'''
    factor = numpy.real(factor_xz01 * numpy.conj(factor_xz01))
    beta = numpy.linspace(0, 180, nbeta)
    import matplotlib.pyplot as plt
    plt.plot(beta, factor, label="nStructureFactor")
    plt.xlabel('beta')
    plt.ylabel('|G00|$^{2}$')
    plt.title("O2")
    plt.xlim([0, 180])
    plt.ylim(bottom=0)
    plt.show()