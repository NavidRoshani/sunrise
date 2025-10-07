import sunrise as sun
from math import pi
from sunrise.graphical import *

circuit = GraphicalCircuit([
    SingleExcitation(0, 2, angle="a"),
    SingleExcitation(0, 2, angle="b"),
    SingleExcitation(0, 2, angle="c"),
    SingleExcitation(0, 2, angle="d"),
])
circuit.export_qpic(filename="before_assignment")
# sun.qpic_to_pdf("all_gates_example")
circuit.export_to("before_assignment.pdf")
variables = {"a": 0, "b": pi / 4, "c": pi / 2, "d": pi}
new_circuit = circuit.map_variables(variables)
new_circuit.export_qpic(filename="after_assignment")
new_circuit.export_to("after_assignment.pdf")