import tequila as tq
import pytest
import sunrise as sn
from sunrise.expval import INSTALLED_FERMIONIC_BACKENDS,Braket
from numpy import isclose



HAS_TCC = "tcc" in INSTALLED_FERMIONIC_BACKENDS
HAS_FQE = "fqe" in INSTALLED_FERMIONIC_BACKENDS


@pytest.mark.parametrize("geom",["H 0.0 0.0 0.0\nH 0.0 0.0 1.6\nH 0.0 0.0 3.2\nH 0.0 0.0 4.8","H 0. 0. 0.\n Be 0. 0. 1.6\n H 0. 0. 3.2"])
@pytest.mark.parametrize('backend',INSTALLED_FERMIONIC_BACKENDS)
def test_spa(geom,backend):
    mol = tq.Molecule(geometry=geom,basis_set='sto-3g',transformation='reordered-jordan-wigner').use_native_orbitals()
    edges = sn.Molecule(geometry=geom,basis_set='sto-3g').get_spa_edges()
    U = mol.make_ansatz("SPA",edges=edges,optimize=False)
    circuit = sn.FCircuit.from_edges(edges=edges,n_orb=mol.n_orbitals)
    expval = tq.ExpectationValue(H=mol.make_hamiltonian(),U=U)
    sunval = Braket(molecule=mol,circuit=circuit,backend=backend)
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
    circuit = sn.FCircuit.from_Qcircuit(U)
    expval = tq.ExpectationValue(H=mol.make_hamiltonian(),U=U)
    sunval = Braket(molecule=mol,circuit=circuit,backend=backend)
    tqE = tq.minimize(expval,silent=True)
    sunval.minimize()
    sunE = sunval.energy
    tqwfn = tq.simulate(U,tqE.angles)
    sunwfn = tq.simulate(U,sunval.variables)
    assert isclose(tqE.energy,sunE)
    assert isclose(abs(tqwfn.inner(sunwfn)),1,1.e-3)

@pytest.mark.parametrize('backend',INSTALLED_FERMIONIC_BACKENDS)
def test_transition(backend):
    geom = 'H 0. 0. 0. \n H 0. 0. 1. \n H 0. 0. 2. \n H 0. 0. 3.'
    mol = tq.Molecule(geometry=geom,basis_set='sto-3g',transformation='reordered-jordan-wigner').use_native_orbitals()
    H = mol.make_hamiltonian()
    U1 = mol.make_ansatz("SPA",edges=[(0,1),(2,3)])
    U2 = mol.make_ansatz("SPA",edges=[(0,2),(1,3)])
    res1 = tq.minimize(tq.ExpectationValue(H=H,U=U1),silent=True)
    res2 = tq.minimize(tq.ExpectationValue(H=H,U=U2),silent=True)
    bra = sn.FCircuit.from_edges([(0,1),(2,3)],n_orb=mol.n_orbitals)
    ket = sn.FCircuit.from_edges([(0,2),(1,3)],n_orb=mol.n_orbitals)
    rov,iov = tq.BraKet(bra=U1,ket=U2,H=H)
    res1.angles.update(res2.angles)
    tqov = tq.simulate(rov,variables=res1.angles) + tq.simulate(iov,variables=res1.angles)
    sunval = Braket(molecule=mol,bra=bra,ket=ket,backend=backend)
    bkov = sunval.simulate(res1.angles)
    assert isclose(tqov,bkov,atol=1.e-3)