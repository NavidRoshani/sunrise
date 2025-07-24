from tequila.quantumchemistry.qc_base import QuantumChemistryBase
import pyscf

def from_tequila(molecule:QuantumChemistryBase,**kwargs)->pyscf.gto.Mole:
    geometry = molecule.parameters.get_geometry()
    pyscf_geomstring = ""
    for atom in geometry:
        pyscf_geomstring += "{} {} {} {};".format(atom[0], atom[1][0], atom[1][1], atom[1][2])

    if "point_group" in kwargs:
        point_group = kwargs["point_group"]
    else:
        point_group = None

    mol = pyscf.gto.M(
        atom=pyscf_geomstring,
        basis=molecule.parameters.basis_set,
        charge=molecule.parameters.charge,
        verbose=0,
        spin=0,
    )
    if point_group is not None:
        if point_group.lower() != "c1":
            mol.symmetry = True
            mol.symmetry_subgroup = point_group
        else:
            mol.symmetry = False
    else:
        mol.symmetry = True
    mol.symmetry = False

    mol.build(parse_arg=False)
    mol.ao2mo(mo_coeffs=molecule.integral_manager.orbital_coefficients)
    mol.build()
    # solve restricted HF
    mf = pyscf.scf.RHF(mol)#.run()
    mf.verbose = False
    if "verbose" in kwargs:
        mf.verbose = kwargs["verbose"]
    mf.mo_coeff = molecule.integral_manager.orbital_coefficients
    # mf.kernel()
    
    return mf