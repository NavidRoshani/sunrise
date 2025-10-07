import tequila as tq
import sunrise as sun

mol = tq.Molecule(geometry="H 0. 0. 0. \n H 0. 0. 1.",basis_set="sto-3g")
U = tq.QCircuit()
U += tq.gates.Y(2)
U += mol.make_excitation_gate(indices=[(0,2),(1,3)],angle="a")
U += mol.make_excitation_gate(indices=[(4,6)],angle="b")
U += tq.gates.QubitExcitation(target=[5,7],angle="c")
U += tq.gates.Trotterized(generator=mol.make_excitation_generator(indices=[(0,2)]),angle="d")
U += mol.make_excitation_gate(indices=[(4,6)],angle="b")
U += mol.make_excitation_gate(indices=[(5,7)],angle="b")
U += mol.UR(0,1,1)
U += mol.UC(1,2,2)

visual_circuit = sun.graphical.GraphicalCircuit.from_circuit(U=U, n_qubits_is_double=True)

visual_circuit.export_qpic("from_circuit_example") # Create qpic file
visual_circuit.export_to("from_circuit_example.pdf") # Create pdf file
# visual_circuit.export_to("from_circuit_example.png") # Create png file