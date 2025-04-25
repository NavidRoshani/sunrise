import fqe as fqe
import openfermion
import tequila as tq
import numpy
import time
## created https://github.com/JdelArco98/tequila.git with the modifications Jakob said
def compile_circuit(circuit):
    fc = []
    for gate in circuit.gates:
        if isinstance(gate, tq.gates.QubitExcitationImpl):
            p = gate.parameter.name
            i,j = p[0][0]
            idx = [(2*i,2*j),(2*i+1,2*j+1)]
        if isinstance(gate, tq.quantumchemistry.qc_base.FermionicGateImpl):
            p = gate.parameter.name
            idx = p[0]
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
U = mol.make_ansatz(name="UpCCD", hcb_optimization=False, include_reference=False)
H = mol.make_hamiltonian()
E = tq.ExpectationValue(H=H, U=mol.prepare_reference()+U)
result = tq.minimize(E, silent=True)
variables = result.variables

HF = mol.make_molecular_hamiltonian()
HH = openfermion.transforms.get_fermion_operator(HF)
HH = fqe.get_hamiltonian_from_openfermion(HH,mol.n_orbitals)
FU = compile_circuit(U)
wfn = fqe.Wavefunction([[mol.n_electrons, 0, mol.n_orbitals]])
wfn.set_wfn(strategy="hartree-fock")

EX = tq.compile(E)
start = time.time()
EX(result.variables)
end = time.time()
print("took {}s".format(end-start))

start = time.time()
variables = [variables[gate.parameter] for gate in U.gates]
for i,g in enumerate(FU):
    wfn = wfn.time_evolve(-0.5*variables[i], g)
energy = wfn.expectationValue(HH)
end = time.time()
print("took {}s".format(end-start))

wfn = fqe.Wavefunction([[mol.n_electrons, 0, mol.n_orbitals]])
wfn.set_wfn(strategy="hartree-fock")
EY = tq.compile(E, backend="fqe",  wfn=wfn, FH=HH, FU=FU, vnames=U.extract_variables())
energy2 = EY(result.variables)
energy2 = EY(result.variables)

result2 = tq.minimize(E, gradient="2-point",  backend="fqe", wfn=wfn, FH=HH, FU=FU, vnames=U.extract_variables())


print(result2.energy)
print(energy2)
print(energy)
#print(result.energy)