import sunrise as sun
from math import pi

circuit = sun.Circuit([
    sun.SingleExcitation(0, 2, angle="a"),
    sun.SingleExcitation(0, 2, angle="b"),
    sun.SingleExcitation(0, 2, angle="c"),
    sun.SingleExcitation(0, 2, angle="d"),
])
sun.export_qpic(circuit, filename="before_assignment", color_range=True, mark_parametrized_gates=True)
variables = {"a": 0, "b": pi / 4, "c": pi / 2, "d": pi}
new_circuit = circuit.map_variables(variables)
sun.export_qpic(new_circuit, filename="after_assignment", color_range=True, mark_parametrized_gates=True)
