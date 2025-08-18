Repository to collect as an extension to [TEQUILA](https://github.com/tequilahub) many chemistry-related projects.

# Installation
Will work on OSX and Linux (no PySCF on Windows)

Install this program with all the dependencies like this:

```bash
conda create -n myenv python=3.10
conda activate myenv

cd project-sunrise
pip install -e .
```


# [Hybrid Molecule](https://doi.org/10.1088/2058-9565/adbdee)
Create your Hybrid Molecule. \
Example :

```python
import sunrise as sun # first time might take some seconds
import tequila as tq

molecule  = sun.Molecule(geometry="H 0. 0. 0. \n Li 0. 0. 1.5",basis_set="sto-3g",select="BBFBF")
print(molecule.select)
```

it can be also initialized as

```python
import sunrise as sun
import tequila as tq
molecule  = sun.Molecule(geometry="H 0. 0. 0. \n Li 0. 0. 1.5",basis_set="sto-3g",select={2:"F",4:"F"})
print(molecule.select)
```

### Construct your circuit
The SPA circuit (and all the automatically built circuits) are already adapted to your encoding

```python
Uspa = molecule.make_ansatz("SPA",edges=[(0,1)])
```
however, one can also  build its own circuits:

```python
U = tq.QCircuit() # see more on https://github.com/tequilahub/tequila-tutorials/blob/main/BasicUsage.ipynb
U += molecule.prepare_reference() # Prepares the reference HF state if any other provided
U += molecule.UC(0,2,angle=(0,2,"a")) #Paired 2e excitation from MO 0 to MO 2
U += molecule.UR(2,4,angle=(2,4,"UR")) # Two One-electron excitation: MO 2_up->4_up + 2_down->4_down TAKE CARE ENCODING
U += molecule.make_excitation_gate(indices=[(0,4),(1,8)],angle=tq.Variable('a')) #Generic excitation

```
### Minimize the Energy of the Circuit Expectation Value

```python
H = molecule.make_hamiltonian() # The molecular Hamiltonian for a given Encoding is automatically built. For custom Hamiltonians please check tutorial above
exp = tq.ExpectationValue(H=H,U=U) #Create the Expectation Value Object
mini = tq.minimize(objective=exp,silent=False,initial_values={}) #Then you minimize the energy. You can provide initial variables
print('Minimized Angles:\n',mini.angles)
print('Minimized Energy: ', mini.energy)
```

### Optimize your Orbitals
Molecular orbitals can be optimized taking advantage of this Hybrid Encoding

```python
result = molecule.optimize_orbitals(molecule=molecule,circuit=Uspa,initial_guess='random') #Since random guess, may take some time
omol = result.molecule
print("Opt SPA Energy = ",result.energy)
print("Select: ",omol.select)
```

Find this example in the test file

# [HCB measurement optimization](https://arxiv.org/abs/2504.03019)
Optimize the measurement procedure of a molecule energy by using the HCB encoding and multiple rotations.

Example using quantum circuit (Scenario II):

```python
import tequila as tq
import sunrise as sun
import numpy as np

# Create the molecule
mol = tq.Molecule(geometry="h 0.0 0.0 0.0\nh 0.0 0.0 1.5\nh 0.0 0.0 3.0\nh 0.0 0.0 4.5", basis_set="sto-3g").use_native_orbitals()
fci = mol.compute_energy("fci")
H = mol.make_hamiltonian()

# Create circuit
U0 = mol.make_ansatz(name="SPA", edges=[(0,1),(2,3)])
UR1 = mol.UR(0,1,angle=np.pi/2) + mol.UR(2,3,angle=np.pi/2) + mol.UR(0,3,angle=-0.2334) + mol.UR(1,2,angle=-0.2334)

UR2 = mol.UR(1,2,angle=np.pi/2) + mol.UR(0,3,angle=np.pi/2)
UR2+= mol.UR(0,1,angle="x") + mol.UR(0,2,angle="y") + mol.UR(1,3,angle="xx") + mol.UR(2,3,angle="yy") + mol.UR(1,2,angle="z") + mol.UR(0,3,angle="zz")
UC2 = mol.UC(1,2,angle="b") + mol.UC(0,3,angle="c")
U = U0 + UR1.dagger() + UR2 + UC2 + UR2.dagger()

variables = {((0, 1), 'D', None): -0.644359150621798, ((2, 3), 'D', None): -0.644359150621816, "x": 0.4322931478168998, "y": 4.980327764918099e-14, "xx": -3.07202211218271e-14, "yy": 0.7167447375727501, "z": -3.982666230146327e-14, "zz": 1.2737831353027637e-13, "c": -0.011081251246998072, "b": 0.49719805420976604}
E = tq.ExpectationValue(H=H, U=U)
full_energy = tq.simulate(E, variables=variables)
print(f"Energy error: {(full_energy-fci)*1000:.2f} mE_h\n")

# Create rotators
graphs = [
    [(0,1),(2,3)],
    [(0,3),(1,2)],
    [(0,2),(1,3)]
]
rotators = []
for graph in graphs:
    UR = tq.QCircuit()
    for edge in graph:
        UR += mol.UR(edge[0], edge[1], angle=np.pi/2)
    rotators.append(UR)

# Apply the measurement protocol
result = sun.rotate_and_hcb(molecule=mol, circuit=U, variables=variables, rotators=rotators, target=full_energy, silent=False)
print(result) # the list of HCB molecules to measure and the residual element discarded

# Compute the energy
energy = 0
for i,hcb_mol in enumerate(result[0]):
    expval = tq.ExpectationValue(U=U+rotators[i], H=hcb_mol.make_hamiltonian())
    energy += tq.simulate(expval, variables=variables)

print(f"Energy of the accumulated HCB contributions: {energy}")
print(f"Error: {energy-full_energy}")
```

Example using a wavefunction (Scenario I):

```python
import tequila as tq
import sunrise as sun
import numpy as np
import openfermion as of
import scipy

# Create the molecule
mol = tq.Molecule(geometry="h 0.0 0.0 0.0\nh 0.0 0.0 1.5\nh 0.0 0.0 3.0\nh 0.0 0.0 4.5", basis_set="sto-3g").use_native_orbitals()
fci = mol.compute_energy("fci")
H = mol.make_hamiltonian()

# Create true wave function
Hof = H.to_openfermion()
Hsparse = of.linalg.get_sparse_operator(Hof)
v,vv = scipy.sparse.linalg.eigsh(Hsparse, sigma=fci)
wfn = tq.QubitWaveFunction.from_array(vv[:,0])
energy = wfn.inner(H * wfn).real

# Create rotators
graphs = [
    [(0,1),(2,3)],
    [(0,3),(1,2)],
    [(0,2),(1,3)]
]
rotators = []
for graph in graphs:
    UR = tq.QCircuit()
    for edge in graph:
        UR += mol.UR(edge[0], edge[1], angle=np.pi/2)
    rotators.append(UR)

# Apply the measurement protocol
result = sun.rotate_and_hcb(molecule=mol, rotators=rotators, target=fci, initial_state=wfn, silent=False)
print(result) # the list of HCB molecules to measure and the residual element discarded

# Compute the energy
energy = 0
for i,hcb_mol in enumerate(result[0]):
    expval = tq.ExpectationValue(U=rotators[i], H=hcb_mol.make_hamiltonian())
    energy += tq.simulate(expval, initial_state=wfn)

print(f"Energy of the accumulated HCB contributions: {energy}")
print(f"Error: {energy-fci}")
```

# Hybridization
Automated framework for the generation of ansatz-specific optimized molecular orbitals. Given the Molecule Geometry 
it identifies atomic hybridization states in order to construct an orbital coefficient matrix and generate a edge 
list of electron pairings and bond assignments.

```python
import sunrise as sun
import tequila as tq

xyzgeometry = """
    3
    Water
    O 0.000000 0.000000 0.000000
    H 0.757000 0.586000 0.000000
    H -0.757000 0.586000 0.000000
    """
graph = sun.Graph.parse_xyz(xyzgeometry)

edges = graph.get_spa_edges()
initial_guess = graph.get_orbital_coefficient_matrix()
geometry = graph.get_geometry_string()

mol = tq.Molecule(geometry=geometry, basis_set='sto-3g', frozen_core=False).use_native_orbitals()
U = mol.make_ansatz(name="HCB-SPA", edges=edges)
opt = tq.chemistry.optimize_orbitals(molecule=mol, circuit=U, initial_guess=initial_guess.T, use_hcb=True)
```
You can check the resulting orbitals with the plot_MO function. See bellow

**Currently implementation works only for sto-3g and s,p orbitals**

# plot_MO
Interface with the [PYSCF](https://pyscf.org/)  [cubegen](https://pyscf.org/pyscf_api_docs/pyscf.tools.html#module-pyscf.tools.cubegen) tool. It generates the orbital
'.cube' files. They may be visualized with many chemical visualizing programs such as [VESTA](https://jp-minerals.org/vesta/en/) or [Avogadro](https://www.openchemistry.org/projects/avogadro2/).

```python
mol = opt.molecule
sun.plot_MO(molecule=mol,file_name="water")
```
The cubefile generation may take some time. Here we provided a tq.Molecule but it also accepts any molecule class with
mol.parameters and mol.integral_manager

#  Circuit Visualizer
Improved circuit visualizer which creates the circuit qpic file with improved circuit structures in common chemistry
building blocks as the electronic excitation gates. It creates gates in molecular orbitals picture, halving the 
number of qubits displayed.

### Automatized way
```python
import tequila as tq
import sunrise as sun

mol = tq.Molecule(geometry="H 0. 0. 0. \n H 0. 0. 1.",basis_set="sto-3g")
U = tq.QCircuit()
U += tq.gates.Y(2) # Generic gate
U += mol.make_excitation_gate(indices=[(0,2),(1,3)],angle="a") # Double excitation
U += mol.make_excitation_gate(indices=[(4,6)],angle="b") # Single excitation
U += tq.gates.QubitExcitation(target=[5,7],angle="c") # Qubit excitation
U += tq.gates.Trotterized(generator=mol.make_excitation_generator(indices=[(0,2)]),angle="d") # Trotterized rotation
U += mol.make_excitation_gate(indices=[(4,6)],angle="b")
U += mol.make_excitation_gate(indices=[(5,7)],angle="b") # Paired single excitation
U += mol.UR(0,1,1) # Orbital rotator
U += mol.UC(1,2,2) # Pair correlator

visual_circuit = sun.from_circuit(U, n_qubits_is_double=True) # Translate tq.QCircuit in renderable Circuit

visual_circuit.export_qpic("from_circuit_example") # Create qpic file
sun.qpic_to_png("from_circuit_example") # Create png file
sun.qpic_to_pdf("from_circuit_example") # Create png file
```

### Manual way
```python
import tequila as tq
import sunrise as sun

circuit = sun.Circuit([
    # Initial state gate, qubits are halved 
    sun.GenericGate(U=tq.gates.X([0,1,2,3]), name="initialstate", n_qubits_is_double=True),

    # Singe excitation, i,j correspond to Spin Orbital index --> ((i,j))
    sun.SingleExcitation(i=1,j=7,angle=1),

    # Double excitation, i,j,k,l correspond to Spin Orbital index --> ((i,j),(k,l))
    sun.DoubleExcitation(i=0,j=4,k=1,l=7,angle=2),

    # Generic gate in the middle of the circuit, qubits are not halved
    sun.GenericGate(U=tq.gates.Y([0, 3]), name="simple", n_qubits_is_double=False),

    # Orbital rotator (double single-excitation), i,j correspond to Molecular Orbital index --> ((2*i,2*j),(2*i+1,2*j+1))
    sun.OrbitalRotatorGate(i=0,j=1,angle=3),

    # Pair correlator (paired double excitation), i,j correspond to Molecular Orbital index --> ((2*i,2*j),(2*i+1,2*j+1))
    sun.PairCorrelatorGate(i=1,j=3,angle=4)
])

circuit.export_qpic("test") # Create qpic file
sun.qpic_to_png("test") # Create png file
sun.qpic_to_pdf("test") # Create pdf file
```

this cirucit can be latter exported to tequila circuit:

```python
U = circuit.construct_circuit()
U += tq.gates.X(1)
```
  