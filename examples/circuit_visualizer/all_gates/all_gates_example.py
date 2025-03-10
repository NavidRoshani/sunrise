import tequila as tq
import src.sunrise.molecularcircuitvisualizer as vs

circuit = vs.Circuit([
    # Initial state gate, qubits are halved 
    vs.GenericGate(U=tq.gates.X([0,1,2,3]), name="initialstate", n_qubits_is_double=True),

    # Singe excitation, i,j correspond to Spin Orbital index --> ((i,j))
    vs.SingleExcitation(i=1,j=7,angle=1),

    # Double excitation, i,j,k,l correspond to Spin Orbital index --> ((i,j),(k,l))
    vs.DoubleExcitation(i=0,j=4,k=1,l=7,angle=2),

    # Generic gate in the middle of the circuit, qubits are not halved
    vs.GenericGate(U=tq.gates.Y([0, 3]), name="simple", n_qubits_is_double=False),

    # Orbital rotator (double single-excitation), i,j correspond to Molecular Orbital index --> ((2*i,2*j),(2*i+1,2*j+1))
    vs.OrbitalRotatorGate(i=0,j=1,angle=3),

    # Pair correlator (paired double excitation), i,j correspond to Molecular Orbital index --> ((2*i,2*j),(2*i+1,2*j+1))
    vs.PairCorrelatorGate(i=1,j=3,angle=4)
])

circuit.export_qpic("all_gates_example") # Create qpic file
vs.qpic_to_png("all_gates_example") # Create png file
vs.qpic_to_pdf("all_gates_example") # Create pdf file