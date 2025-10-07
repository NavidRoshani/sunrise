import tequila as tq
import sunrise as sun
from sunrise.graphical import *


select = {0:"B",1:"F",2:"F",3:"B"}

circuit = GraphicalCircuit([
    # Initial state gate, qubits are halved
    GenericGate(U=tq.gates.X([0,1,2,3]), name="initialstate", n_qubits_is_double=True),

    # Singe excitation, i,j correspond to Spin Orbital index --> ((i,j))
    SingleExcitation(i=1,j=7,angle=1),

    # Double excitation, i,j,k,l correspond to Spin Orbital index --> ((i,j),(k,l))
    DoubleExcitation(i=0,j=4,k=1,l=7,angle=2),

    # Generic gate in the middle of the circuit, qubits are not halved
    GenericGate(U=tq.gates.Y([0, 3]), name="simple", n_qubits_is_double=False),

    # Orbital rotator (double single-excitation), i,j correspond to Molecular Orbital index --> ((2*i,2*j),(2*i+1,2*j+1))
    OrbitalRotatorGate(i=0,j=1,angle=3),

    # Pair correlator (paired double excitation), i,j correspond to Molecular Orbital index --> ((2*i,2*j),(2*i+1,2*j+1))
    PairCorrelatorGate(i=1,j=3,angle=4)
])

circuit.export_qpic("hybrid_molecule",select=select) # Create qpic file
# sun.qpic_to_png("hybrid_molecule") # Create png file
circuit.export_to("hybrid_molecule.pdf") # Create pdf file
