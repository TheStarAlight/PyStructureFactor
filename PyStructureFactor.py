import numpy
import math
import multiprocessing as mp
from multiprocessing.dummy import Pool as ThreadPool
from pyscf import df, scf
from scipy import special
from pyscf import gto, lib, dft
from pyscf.dft import radi
from wigner import wigner_dl

def convert_sph2cart(xyz):
    """
    Transforms the spherical coordinates to Cartesian coordinates.

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
    """
    To find the index of HOMO orbit according to the orbital occupation. Returns i-1, denoting i the first orbit whose occupancy number is zero.

    Parameters:
        np1: Orbital occupancy number, array.

    Returns:
        The index of HOMO orbit starting from 0.
    """
    for i in range(np1.size):
        if np1[i] == 0:
            return i - 1


def vc_orbit(mol, z, coeff, coords, dm, hf_method='RHF'):
    """To calculate the core potential times the ionizing orbital wave function, that is, V_c ψ_0.

    Parameters:
        mol: Mole
            The molecule object.
        z: int
            The charge of the parent ion.
        coeff: 2D array, shape (:,index)
            Molecular orbital coefficient of the ionizing orbit.
        coords: 2D array, shape (N,3)
            Cartesian grid points(x,y,z)
        dm: ndarray
            Density matrix of molecule.
        hf_method : str
            Indicates whether 'RHF' or 'UHF' should be used in molecular HF calculation. Default is 'RHF'.

    Returns:
        The core potential oprator times the orbital wave function.

    """
    ngrids = coords.shape[0]
    vnuc_r = z / numpy.einsum('xi,xi->x', coords, coords) ** .5
    vnuc = 0
    for i in range(mol.natm):
        r = mol.atom_coord(i)
        z = mol.atom_charge(i)
        rp = r - coords
        vnuc += z / numpy.einsum('xi,xi->x', rp, rp) ** .5
    if hf_method == 'UHF':
        dm_vex = dm[0]
        dm_vele = dm[1] + dm[0]
    elif hf_method == 'RHF':
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


def omega_wvl(m, n_xi, kappa, z, l):
    """
    The normalization coefficient of Omega.

    Parameters
        m: int
            Magnetic quantum number, should satisfy `l >= m >= -l`.
        l: int
            Angular quantum number, should be a non-negative integer.
        n_ep: int
            Parabolic quantum number, should be a non-negative integer.
        z: int
            The charge of the parent ion.
        kappa: float
            `sqrt(-2*E0)`, E0 is orbital energy.
    Returns:
        The normalization coefficient of Omega for the given parameter.
    """
    part1 = ((-1) ** (l + (abs(m) - m) / 2 + 1)) * (2 ** (l + 3 / 2)) * (
            kappa ** (z / kappa - (abs(m) + 1) / 2 - n_xi))
    part2 = math.sqrt(
        (2 * l + 1) * math.factorial(l + m) * math.factorial(l - m) * math.factorial(
            abs(m) + n_xi) * math.factorial(
            n_xi)) * math.factorial(l) / math.factorial(2 * l + 1)
    range_len = min(n_xi, l - abs(m))
    part3 = 0

    for k in range(range_len + 1):
        part3 += special.gamma(l + 1 - z / kappa + n_xi - k) / (
                math.factorial(k) * math.factorial(l - k) * math.factorial(abs(m) + k) * math.factorial(
            l - abs(m) - k) * math.factorial(n_xi - k))
    omega_norm = part1 * part2 * part3
    return omega_norm


def omega_r(kappa, z, l, coords):
    """
    Main part of Omega excluding the normalization coefficient. Will calculate all m' in [-l,l].

    # Parameters:
        l : int
            Angular quantum number, should be a non-negative integer.
        z : int
            The charge of the parent ion.
        kappa : float
            `sqrt(-2*E0)`, E0 is orbital energy.

    Returns:
        An numpy array of shape(2l+1, len(coords)).
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

def get_structure_factor(mol,
                        rel_homo_index  = 0,
                        channel         = (0,0),
                        lmax            = 10,
                        hf_method       = 'RHF',
                        atom_grid_level = 3,
                        orient_grid_size= (90,1),
                        move_dip_zero   = True,
                        rmax            = 40):
    """
    Calculates the molecular structure factor $G$ according to the given parameters.

    # Parameters:
        mol : Mole
            The molecule object. Initialized by invoking `pyscf.M` or `pyscf.gto.M`.
        rel_homo_index : int
            Index of the ionizing orbital relative to the HOMO. Default is 0. e.g., HOMO -> 0, LUMO -> +1, HOMO-1 -> -1, ...
        channel : tuple
            Parabolic channel ν=(nξ, m). Default is (0,0).
        lmax : int
            The maximum angular quantum number (larger l would be cut off) used in the sum. Default is 10.
        hf_method : str
            Indicates whether 'RHF' or 'UHF' should be used in molecular HF calculation. Default is 'RHF'. [!] Note: Must use 'UHF' for multiplet molecules.
        atom_grid_level : int
            Level of fineness of the grid used in integration (see also pyscf.dft.Grid), which controls the number of radial and angular grids, ranging from 0 to 9. Default is 3.
        orient_grid_size : tuple
            Indicates the size of (β,γ) grid (in the output) in β,γ directions respectively. Default is (90,1). The grid is uniform, with β ranging from [0,π) and γ ranging from [0,2π).
        move_dip_zero : bool
            Indicates whether to move the molecule so that the dipole of the parent ion equals zero. Default true.
        rmax : float
            Indicates the cut off limit of the radial grid points, points of radius>rmax would not be accounted in calculation. Default is 40.

    # Returns:
        A numpy array containing the structure factors $G$ on the (β,γ) orientation grid. Shape = orient_grid_size.
    """

    Z = mol.charge + 1
    n_xi = channel[0]
    m = channel[1]
    n_beta = orient_grid_size[0]
    n_gamma = orient_grid_size[1]


    if hf_method == 'RHF':
        mf = scf.RHF(mol).run(verbose=0)
        mo_occ = mf.mo_occ
        orbit_energy = mf.mo_energy
        coeff = mf.mo_coeff
    elif hf_method == 'UHF':
        mf = scf.UHF(mol).run(verbose=0)
        mo_occ = mf.mo_occ[0]
        orbit_energy = mf.mo_energy[0]
        coeff = mf.mo_coeff[0]
    else:
        raise Exception("error: method must be either 'RHF' or 'UHF'")

    index = get_homo_index(mo_occ) + rel_homo_index
    energy_index = orbit_energy[index]
    coeff = numpy.expand_dims(coeff[:, index], axis=0)
    kappa = math.sqrt(-2 * energy_index)
    dm = mf.make_rdm1()

    if move_dip_zero:
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
        dip_moment = mf.dip_moment(unit="A.U.",verbose=0)
        D = (u - dip_moment)
        move_distance = D/Z
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


    g = dft.Grids(mol)
    g.level = atom_grid_level
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

    uz1 = orbital_dip(coeff, mol)[2]
    vc_ionorbit = vc_orbit(mol, Z, coeff, coords, dm, hf_method=hf_method)

    I = [None]*(lmax+1)   # I[l][l+m'] stores the integrals I_{lm'}^{\nu}
    for l in range(abs(m), lmax + 1):
        wvl = omega_wvl(m, n_xi, kappa, Z, l)
        omega = omega_r(kappa, Z, l, coords) * wvl * weights
        I[l] = numpy.dot(omega, vc_ionorbit)
    def factor_G(beta,gamma):
        uz = uz1 * numpy.cos(beta)
        Gsum = numpy.complex128(0.0)
        for l in range(abs(m), lmax + 1):   # l  in |m|:lmax
            for mp in range(-l,l+1):        # m' in -l:l
                Gsum += I[l][l+mp] * wigner_dl(l,l,m,mp,beta)[0] * numpy.exp(-1j*mp*gamma)
        return Gsum * numpy.exp(-kappa * uz)

    beta_grid = numpy.linspace(0, math.pi, n_beta)
    gamma_grid = numpy.linspace(0, 2*math.pi, n_gamma)
    G_grid = numpy.zeros(shape=(n_beta,n_gamma), dtype=numpy.complex128)
    for ibeta in range(n_beta):
        for igamma in range(n_gamma):
            G_grid[ibeta,igamma] = factor_G(beta_grid[ibeta],gamma_grid[igamma])
    return G_grid
