
import numpy as np
import scipy.optimize
from matplotlib import pyplot as plt
from scipy.optimize import minimize
import tequila as tq
import pytest
import sunrise as sn
import sunrise.expval
from sunrise.expval import INSTALLED_FERMIONIC_BACKENDS,Braket
from numpy import isclose
import random
from datetime import datetime



HAS_TCC = "tcc" in INSTALLED_FERMIONIC_BACKENDS
HAS_FQE = "fqe" in INSTALLED_FERMIONIC_BACKENDS

#todo: Fix tequila backend to be consistent with the others
@pytest.mark.parametrize("geom",["H 0.0 0.0 0.0\nH 0.0 0.0 1.6\nH 0.0 0.0 3.2\nH 0.0 0.0 4.8","H 0. 0. 0.\n Be 0. 0. 1.6\n H 0. 0. 3.2"])
@pytest.mark.parametrize('backend',INSTALLED_FERMIONIC_BACKENDS)
def test_spa(geom,backend):
    if backend == "tequila":
        pytest.skip("Skipping tequila")
    mol = tq.Molecule(geometry=geom,basis_set='sto-3g',transformation='reordered-jordan-wigner').use_native_orbitals()
    edges = sn.Molecule(geometry=geom,basis_set='sto-3g',nature='hybrid').get_spa_edges()
    U = mol.make_ansatz("SPA",edges=edges)
    circuit = sn.FCircuit.from_edges(edges=edges,n_orb=mol.n_orbitals)
    expval = tq.ExpectationValue(H=mol.make_hamiltonian(),U=U)
    sunval = Braket(molecule=mol,ket=circuit,backend=backend)
    tqE = tq.minimize(expval,silent=True)
    e = sn.minimize(sunval, silent=True)
    sunE = e.energy

    tqwfn = tq.simulate(U,tqE.angles)
    sunwfn = sn.simulate(U,e.variables)
    assert isclose(tqE.energy,sunE)
    assert isclose(abs(tqwfn.inner(sunwfn)),1,1.e-3)


@pytest.mark.parametrize("geom",["H 0.0 0.0 0.0\nH 0.0 0.0 1.6\nH 0.0 0.0 3.2\nH 0.0 0.0 4.8","H 0. 0. 0.\n Be 0. 0. 1.6\n H 0. 0. 3.2"])
@pytest.mark.parametrize('backend',INSTALLED_FERMIONIC_BACKENDS)
def test_upccsd(geom,backend):
    if backend == "tequila":
        pytest.skip("Skipping tequila")
    mol = tq.Molecule(geometry=geom,basis_set='sto-3g',transformation='reordered-jordan-wigner')
    U = mol.make_ansatz("UpCCSD")
    fmol = sn.Molecule(geometry=geom,basis_set='sto-3g',nature='fermionic')
    circuit = fmol.make_ansatz("UpCCSD")
    expval = tq.ExpectationValue(H=mol.make_hamiltonian(),U=U)
    sunval = Braket(molecule=mol,circuit=circuit,backend=backend)
    tqE = tq.minimize(expval,silent=True)
    snE = sn.minimize(sunval, silent=True)
    assert isclose(tqE.energy,snE.energy)

@pytest.mark.parametrize('backend',INSTALLED_FERMIONIC_BACKENDS)
def test_transition(backend):
    if backend == "tequila":
        pytest.skip("Skipping tequila")
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
    tq_ov = tq.simulate(rov,variables=res1.angles) + tq.simulate(iov,variables=res1.angles)
    sunval = Braket(molecule=mol,bra=bra,ket=ket,backend=backend)
    sn_ov = sunval(res1.angles)
    assert isclose(tq_ov,sn_ov,atol=1.e-3)

@pytest.mark.parametrize("geom",["H 0.0 0.0 0.0\nH 0.0 0.0 1.6\nH 0.0 0.0 3.2\nH 0.0 0.0 4.8","H 0. 0. 0.\n Be 0. 0. 1.6\n H 0. 0. 3.2"])
@pytest.mark.parametrize('backend',INSTALLED_FERMIONIC_BACKENDS)
def test_maped_variables(geom,backend):
    if backend == "tequila":
        pytest.skip("Skipping tequila")
    random.seed(datetime.now().timestamp())
    mol = tq.Molecule(geometry=geom,basis_set='sto-3g',transformation='reordered-jordan-wigner').use_native_orbitals()
    edges = sn.Molecule(geometry=geom,basis_set='sto-3g',nature='hybrid').get_spa_edges()
    U = mol.make_ansatz("SPA",edges=edges)
    mapa = {d:random.random()*np.pi for d in U.extract_variables()}
    U = U.map_variables(mapa)
    circuit = sn.FCircuit.from_edges(edges=edges,n_orb=mol.n_orbitals)
    circuit = circuit.map_variables(mapa)
    expval = tq.ExpectationValue(H=mol.make_hamiltonian(),U=U)
    sunval = Braket(molecule=mol,ket=circuit,backend=backend)
    tqE = tq.simulate(expval,{})
    sunE = sn.simulate(sunval,{})
    assert isclose(tqE,sunE)


#todo remove tests below
@pytest.mark.parametrize("geom", ["H 0.0 0.0 0.0\nH 0.0 0.0 1.6\nH 0.0 0.0 3.2\nH 0.0 0.0 4.8"])
def test_overlap_minimzation_fqe(geom):
    mol = tq.Molecule(geometry=geom, basis_set='sto-3g',transformation='reordered-jordan-wigner').use_native_orbitals()
    H = mol.make_hamiltonian()
    U1 = mol.make_ansatz("SPA", edges=[(0, 1), (2, 3)])
    U2 = mol.make_ansatz("SPA", edges=[(0, 2), (1, 3)])
    bra = sn.FCircuit.from_edges([(0, 1), (2, 3)], n_orb=mol.n_orbitals)
    ket = sn.FCircuit.from_edges([(0, 2), (1, 3)], n_orb=mol.n_orbitals)

    tqbraket = tq.BraKet(bra=U1, ket=U2, H=H)
    both = tq.BraKet(bra=U1, ket=U2, H=H)

    tqE = tq.minimize(tqbraket[0], silent=False)
    # tqov = tq.minimize(rov) + tq.minimize(iov)

    fqebra = sunrise.expval.FQEBraKet(molecule=mol, ket=ket, bra=bra)
    variables = bra.variables
    variables += ket.variables
    variables = list(dict.fromkeys(variables))
    init_vars = {vs: 0 for vs in variables}
    x0 = list(init_vars.values())
    r = minimize(
        fun=fqebra,
        x0=x0,
        jac="2-point",
        method="bfgs",
        options={"finite_diff_rel_step": 1.e-6, "disp": True},
        tol=1.e-10
    )
    sn_angles = {k: v for k, v in zip(variables, r.x)}
    # sunwfn = tq.simulate(U, sn_angles)
    tqov = tq.simulate(both[0], silent=True, variables=sn_angles)
    print(tqov, r.fun.real)
    assert isclose(tqov, r.fun.real, atol=1.e-3)

    assert isclose(tqE.energy, r.fun.real, atol=1.e-3)
    # assert isclose(abs(tqwfn.inner(sunwfn)), 1, 1.e-3)

