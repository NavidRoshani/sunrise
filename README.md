Repository to collect as an extension to [TEQUILA](https://github.com/tequilahub) many chemistry-related projects.

# Installation
Will work on OSX and Linux (no PySCF on Windows)

Install this program with all the dependencies like this:

```bash
conda create -n myenv python=3.9
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

### Construct your circuit circuit
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
