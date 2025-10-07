import tequila as tq
from sunrise.graphical import *


U = GraphicalCircuit()
U  += GenericGate(U=tq.gates.X([0,1,2,3]), name="Generic1", n_qubits_is_double=True)
U  += GenericGate(U=tq.gates.Y([2,3,4,5]), name="Generic2", n_qubits_is_double=True)
U  += GenericGate(U=tq.gates.Y([0,1,4,5]), name="Generic3", n_qubits_is_double=True)
U.export_qpic("simplified_G") # Create qpic file
qpic_to_png("simplified_G") # Create png file, you need pdflatex
qpic_to_pdf("simplified_G") # Create pdf file, you need pdflatex