import sunrise as sun
import tequila as tq
import numpy as np
from fqe_expval import FQEBraKet

mol = tq.Molecule("H 0 0 0\nH 0 0 1\nH 0 0 2\nH 0 0 3", "sto-3g").use_native_orbitals()
H = mol.make_hamiltonian()
c,h,g = mol.get_integrals()
# [[(0,2,1,3),(2,4,3,5)],[(0,2),(1,3)]]

# Single excitation
# (0,4): np.pi/2
angle = np.pi/2
print("teq:", tq.simulate(tq.ExpectationValue(mol.prepare_reference() + mol.make_excitation_gate((0,4), angle), H)))
# print(f"fqe:", FQEBraKet(h, g.elems, c, ket_instructions=[[indices]])({indices:angle}).real)
print("fqe:", FQEBraKet(h, g.elems, c, ket_instructions=[[(0,4)]])({(0,4):angle}).real)
print()

# Double excitation
# (0,4,1,5): np.pi/2
angle = np.pi/2
print("teq:", tq.simulate(tq.ExpectationValue(mol.prepare_reference() + mol.make_excitation_gate((0,4,1,5), angle), H)))
print("teq:", tq.simulate(tq.ExpectationValue(mol.prepare_reference() + mol.make_excitation_gate([(0,4),(1,5)], angle), H)))
print("fqe:", FQEBraKet(h, g.elems, c, ket_instructions=[[(0,4,1,5)]])({(0,4,1,5):angle}).real)
print()

# Paired single excitation
# [(0,4),(1,5)]: np.pi/2
angle = np.pi/2
print("teq:", tq.simulate(tq.ExpectationValue(mol.prepare_reference() + mol.make_excitation_gate((0,4), angle) + mol.make_excitation_gate((1,5), angle), H)))
print("teq:", tq.simulate(tq.ExpectationValue(mol.prepare_reference() + mol.UR(0,2,angle), H)))
print("fqe:", FQEBraKet(h, g.elems, c, ket_instructions=[[(0,4),(1,5)]])({"R":angle}).real)
print()

# Two single excitations with different angles
# (0,4): np.pi/2, (1,5): np.pi/3
angle1 = np.pi/2
angle2 = np.pi/3
print("teq:", tq.simulate(tq.ExpectationValue(mol.prepare_reference() + mol.make_excitation_gate((0,4), angle1) + mol.make_excitation_gate((1,5), angle2), H)))
print("fqe:", FQEBraKet(h, g.elems, c, ket_instructions=[[(0,4)],[(1,5)]])({(0,4):angle1, (1,5):angle2}).real)
print()

# Two double excitations with different angles
# (0,4,1,5): np.pi/2, (0,6,1,7): np.pi/3
angle1 = np.pi/2
angle2 = np.pi/3
print("teq:", tq.simulate(tq.ExpectationValue(mol.prepare_reference() + mol.make_excitation_gate((0,4,1,5), angle1) + mol.make_excitation_gate((0,6,1,7), angle2), H)))
print("fqe:", FQEBraKet(h, g.elems, c, ket_instructions=[[(0,4,1,5)],[(0,6,1,7)]])({(0,4,1,5):angle1, (0,6,1,7):angle2}).real)
print()

# Braket
# single and double
ket = mol.prepare_reference() + mol.make_excitation_gate((0,4), np.pi/2)
bra = mol.prepare_reference() + mol.make_excitation_gate([(0,4),(1,5)], np.pi/3)
print("teq:", tq.simulate(tq.BraKet(ket, bra, H)[0]))
print("fqe:", FQEBraKet(h, g.elems, c, ket_instructions=[[(0,4)]], bra_instructions=[[(0,4,1,5)]])({(0,4):np.pi/2}).real)