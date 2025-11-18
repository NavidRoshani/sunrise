import tequila as tq
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from entropy_utils_qubit import *

# canonical orbitals
mol = tq.Molecule("H 0 0 0\nH 0 0 1.5\nH 0 0 3\nH 0 0 4.5", "sto-3g")
# print(mol.integral_manager.orbital_coefficients)
H = mol.make_hamiltonian()
U = mol.prepare_reference()
print(tq.simulate(U))
print("I_01:", mutual_info_2ordm(mol, U, orb_a=[0,1], orb_b=[2,3]))
print("I_02:", mutual_info_2ordm(mol, U, orb_a=[0,1], orb_b=[4,5]))
print("I_03:", mutual_info_2ordm(mol, U, orb_a=[0,1], orb_b=[6,7]))
print("I_12:", mutual_info_2ordm(mol, U, orb_a=[2,3], orb_b=[4,5]))
print("I_13:", mutual_info_2ordm(mol, U, orb_a=[2,3], orb_b=[6,7]))
print("I_23:", mutual_info_2ordm(mol, U, orb_a=[4,5], orb_b=[6,7]))
print("E_0:", pure_state_entanglement(mol, U, orb_a=[0,1]))
print("E_1:", pure_state_entanglement(mol, U, orb_a=[2,3]))
print("E_2:", pure_state_entanglement(mol, U, orb_a=[4,5]))
print("E_3:", pure_state_entanglement(mol, U, orb_a=[6,7]))
print()

"""
# SPA orbitals
mol = mol.use_native_orbitals() # localized orbitals
H = mol.make_hamiltonian()
U = mol.make_ansatz(name="SPA", edges=[(0,1),(2,3)])
guess = np.eye(4)
guess[0] = [1.0, 1.0, 0.0, 0.0]
guess[1] = [1.0, -1., 0.0, 0.0]
guess[2] = [0.0, 0.0, 1.0, 1.0]
guess[3] = [0.0, 0.0, 1.0, -1.]
opt = tq.chemistry.optimize_orbitals(mol, circuit=U, initial_guess=guess.T, silent=True)
mol = opt.molecule
H = mol.make_hamiltonian()
E = tq.ExpectationValue(U,H)
result = tq.minimize(E, silent=True)
U = U.map_variables(result.variables)
print(tq.simulate(U))
print("I_01:", mutual_info_2ordm(mol, U, orb_a=[0,1], orb_b=[2,3]))
print("I_02:", mutual_info_2ordm(mol, U, orb_a=[0,1], orb_b=[4,5]))
print("I_03:", mutual_info_2ordm(mol, U, orb_a=[0,1], orb_b=[6,7]))
print("I_12:", mutual_info_2ordm(mol, U, orb_a=[2,3], orb_b=[4,5]))
print("I_13:", mutual_info_2ordm(mol, U, orb_a=[2,3], orb_b=[6,7]))
print("I_23:", mutual_info_2ordm(mol, U, orb_a=[4,5], orb_b=[6,7]))
print("E_0:", pure_state_entanglement(mol, U, orb_a=[0,1]))
print("E_1:", pure_state_entanglement(mol, U, orb_a=[2,3]))
print("E_2:", pure_state_entanglement(mol, U, orb_a=[4,5]))
print("E_3:", pure_state_entanglement(mol, U, orb_a=[6,7]))
print()
"""

# localized orbitals
mol = mol.use_native_orbitals()
H = mol.make_hamiltonian()
U = mol.make_ansatz(name="SPA", edges=[(0,1),(2,3)])
guess = np.eye(4)
guess[0] = [1.0, 1.0, 0.0, 0.0]
guess[1] = [1.0, -1., 0.0, 0.0]
guess[2] = [0.0, 0.0, 1.0, 1.0]
guess[3] = [0.0, 0.0, 1.0, -1.]
opt = tq.chemistry.optimize_orbitals(mol, circuit=U, initial_guess=guess.T, silent=True)
UR = mol.get_givens_circuit(opt.mo_coeff)
U += UR.dagger()
E = tq.ExpectationValue(U,H)
result = tq.minimize(E, silent=True)
U = U.map_variables(result.variables)
print(tq.simulate(U))
print("I_01:", mutual_info_2ordm(mol, U, orb_a=[0,1], orb_b=[2,3]))
print("I_02:", mutual_info_2ordm(mol, U, orb_a=[0,1], orb_b=[4,5]))
print("I_03:", mutual_info_2ordm(mol, U, orb_a=[0,1], orb_b=[6,7]))
print("I_12:", mutual_info_2ordm(mol, U, orb_a=[2,3], orb_b=[4,5]))
print("I_13:", mutual_info_2ordm(mol, U, orb_a=[2,3], orb_b=[6,7]))
print("I_23:", mutual_info_2ordm(mol, U, orb_a=[4,5], orb_b=[6,7]))
print("E_0:", pure_state_entanglement(mol, U, orb_a=[0,1]))
print("E_1:", pure_state_entanglement(mol, U, orb_a=[2,3]))
print("E_2:", pure_state_entanglement(mol, U, orb_a=[4,5]))
print("E_3:", pure_state_entanglement(mol, U, orb_a=[6,7]))