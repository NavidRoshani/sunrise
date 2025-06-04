import fqe
import openfermion as of
import tequila as tq
import numpy as np
import time

def compile_circuit(circuit):
    fc = []
    for gate in circuit.gates:
        if isinstance(gate, tq.quantumchemistry.qc_base.FermionicGateImpl):
            idx = gate.indices
        elif isinstance(gate, tq.gates.QubitExcitationImpl):
            p = gate.target
            i,j = p[0],p[1]
            idx = [(i,j),(i+1,j+1)]

        op = mol.make_excitation_generator(indices=idx, fermionic=True)
        fc.append(op)
    return fc

geom = """
H 0.0 0.0 0.0
H 0.0 0.0 1.0
H 0.0 0.0 2.0
H 0.0 0.0 3.0
"""

mol = tq.Molecule(geometry=geom, basis_set="sto-3g")
HF = mol.make_molecular_hamiltonian()
HH = of.transforms.get_fermion_operator(HF)
HH = fqe.get_hamiltonian_from_openfermion(HH,mol.n_orbitals)

U = mol.make_ansatz(name="UpCCSD", hcb_optimization=False, include_reference=False)
U = mol.make_excitation_gate((0,2),"02") + mol.make_excitation_gate((1,3),"13") + mol.make_excitation_gate(((0,2),(1,3)),"02,13") + mol.make_excitation_gate(((0,4),(1,3)),"04,13")
variables = tq.minimize(tq.ExpectationValue(H=mol.make_hamiltonian(), U=mol.prepare_reference()+U), silent=True).variables
print(variables)
FU = compile_circuit(U)
wfn = fqe.Wavefunction([[mol.n_electrons, 0, mol.n_orbitals]])
wfn.set_wfn(strategy="hartree-fock")
for i,g in enumerate(FU):
    print(g)
    wfn = wfn.time_evolve(-0.5*variables[i], g)
energy = wfn.expectationValue(HH)