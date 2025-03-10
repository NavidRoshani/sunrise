from src.sunrise.molecularcircuitvisualizer import qpic_visualization as qp

circuit =[
    qp.SingleExcitation(0, 3, angle="a"),
    qp.DoubleExcitation(0, 2, 1, 3, angle=1.0),
    qp.DoubleExcitation(0, 2, 1, 5, angle="c"),
    qp.DoubleExcitation(0, 2, 5, 3, angle=1.),
    qp.DoubleExcitation(0, 2, 4, 6, angle="e"),
    qp.DoubleExcitation(0, 2, 4, 9, angle=1.),
]
qp.export_to_qpic(circuit, filename="unpaired_example.qpic")