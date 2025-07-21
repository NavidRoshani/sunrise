import tequila as tq
import pytest
from sunrise import Molecule
from sunrise.expval import INSTALLED_FERMIONIC_BACKENDS,Braket
from numpy import isclose



HAS_TCC = "tcc" in INSTALLED_FERMIONIC_BACKENDS
HAS_FQE = "fqe" in INSTALLED_FERMIONIC_BACKENDS



@pytest.mark.parametrize("geom",["H 0.0 0.0 0.0\nH 0.0 0.0 1.6\nH 0.0 0.0 3.2\nH 0.0 0.0 4.8","H 0. 0. 0.\n Be 0. 0. 1.6\n H 0. 0. 3.2"])
@pytest.mark.parametrize('backend',INSTALLED_FERMIONIC_BACKENDS)
def test_spa(geom,backend):
    mol = tq.Molecule(geometry=geom,basis_set='sto-3g').use_native_orbitals()
    hmol = Molecule(geometry=geom,basis_set='sto-3g').use_native_orbitals()
    edges = hmol.get_spa_edges()
    U = mol.make_ansatz("SPA",edges=edges)
    idx,ref = expval.from_Qcircuit(mol.make_ansatz("SPA",edges=edges,optimize=False))
    expval = tq.ExpectationValue(H=mol.make_hamiltonian(),U=U)
    sunval = Braket(molecule=mol,U=idx,reference=ref,backend=backend)
    tqE = tq.minimize(expval,silent=True)
    sunval.minimize()
    sunE = sunval.energy
    tqwfn = tq.simulate(U,tqE.angles)
    sunwfn = tq.simulate(U,sunval.variables)
    assert isclose(tqE.energy,sunE)
    assert isclose(abs(tqwfn.inner(sunwfn)),1)



