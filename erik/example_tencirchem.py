import tencirchem as tcc
import tequila as tq
import numpy as np

mol = tq.Molecule(geometry="H 0 0 0\nH 0 0 1", frozen_core=False, basis_set="sto-3g")#.use_native_orbitals()
fci = mol.compute_energy(method="fci")
# print(fci)
hf = mol.compute_energy(method="hf")
# print(hf)
H = mol.make_hamiltonian()

h = mol.integral_manager.one_body_integrals
g = mol.integral_manager.two_body_integrals
c = mol.integral_manager.constant_term
g = g.reorder('chem')
g = g.elems

U = tcc.UCCSD.from_integral(h, g, mol.n_electrons, c, **{"mo_coeff": mol.integral_manager.orbital_coefficients, "init_method": "zeros", "run_hf": False, "run_mp2": False, "run_ccsd": False, "run_fci": True})

# print(U.params)
# print(U.param_ids)
# print(U.param_to_ex_ops)

# print(U.energy([0,0]))
# print(U.e_fci)
tcc.UCC.get
# print(U.ex_ops)
# U.ex_ops = [...]
U.param_ids = None # decoupled singles
# print(U.param_ids)
# print(U.param_to_ex_ops)

# cost: input parameters return energy(params)
