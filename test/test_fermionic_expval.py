import tequila as tq
import pytest
from sunrise import Molecule
from sunrise.expval import INSTALLED_FERMIONIC_BACKENDS,Braket,from_Qcircuit
from numpy import isclose



HAS_TCC = "tcc" in INSTALLED_FERMIONIC_BACKENDS
HAS_FQE = "fqe" in INSTALLED_FERMIONIC_BACKENDS


@pytest.mark.parametrize("geom",["H 0.0 0.0 0.0\nH 0.0 0.0 1.6\nH 0.0 0.0 3.2\nH 0.0 0.0 4.8","H 0. 0. 0.\n Be 0. 0. 1.6\n H 0. 0. 3.2"])
@pytest.mark.parametrize('backend',INSTALLED_FERMIONIC_BACKENDS)
def test_spa(geom,backend):
    mol = tq.Molecule(geometry=geom,basis_set='sto-3g',transformation='reordered-jordan-wigner').use_native_orbitals()
    edges = Molecule(geometry=geom,basis_set='sto-3g').get_spa_edges()
    U = mol.make_ansatz("SPA",edges=edges,optimize=False)
    idx,ref = from_Qcircuit(U)
    expval = tq.ExpectationValue(H=mol.make_hamiltonian(),U=U)
    sunval = Braket(molecule=mol,indices=idx,reference=tq.simulate(ref,{}),backend=backend,variables=U.extract_variables())
    tqE = tq.minimize(expval,silent=True)
    sunval.minimize()
    sunE = sunval.energy
    tqwfn = tq.simulate(U,tqE.angles)
    sunwfn = tq.simulate(U,sunval.variables)
    assert isclose(tqE.energy,sunE)
    assert isclose(abs(tqwfn.inner(sunwfn)),1,1.e-3)


@pytest.mark.parametrize("geom",["H 0.0 0.0 0.0\nH 0.0 0.0 1.6\nH 0.0 0.0 3.2\nH 0.0 0.0 4.8","H 0. 0. 0.\n Be 0. 0. 1.6\n H 0. 0. 3.2"])
@pytest.mark.parametrize('backend',INSTALLED_FERMIONIC_BACKENDS)
def test_upccsd(geom,backend):
    mol = tq.Molecule(geometry=geom,basis_set='sto-3g',transformation='reordered-jordan-wigner')
    U = mol.make_ansatz("UpCCSD",hcb_optimization=False) #could be hcb-optimized but the from_Qcircuit doesnt work
    idx,ref,variables = from_Qcircuit(U)
    expval = tq.ExpectationValue(H=mol.make_hamiltonian(),U=U)
    sunval = Braket(molecule=mol,indices=idx,reference=tq.simulate(ref,{}),backend=backend,variables=variables)
    tqE = tq.minimize(expval,silent=True)
    sunval.minimize()
    sunE = sunval.energy
    tqwfn = tq.simulate(U,tqE.angles)
    sunwfn = tq.simulate(U,sunval.variables)
    assert isclose(tqE.energy,sunE)
    assert isclose(abs(tqwfn.inner(sunwfn)),1,1.e-3)