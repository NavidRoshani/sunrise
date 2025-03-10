from src.sunrise.molecularcircuitvisualizer import qpic_visualization as qp
from math import pi

if __name__ == '__main__':
    circuit = [
        qp.SingleExcitation(0, 2, angle="a"),
        qp.SingleExcitation(0, 2, angle="b"),
        qp.SingleExcitation(0, 2, angle="c"),
        qp.SingleExcitation(0, 2, angle="d"),
    ]
    qp.export_to_qpic(circuit, filename="before_assignment", color_range=True, mark_parametrized_gates=True)
    new_circuit = []
    variables = {"a": 0, "b": pi / 4, "c": pi / 2, "d": pi}
    for gate in circuit:
        new_circuit.append(gate.map_variables(variables=variables))
    qp.export_to_qpic(new_circuit, filename="after_assignment", color_range=True, mark_parametrized_gates=True)
