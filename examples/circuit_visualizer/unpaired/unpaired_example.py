import sunrise as sun

circuit =sun.GraphicalCircuit([
    sun.SingleExcitation(0, 3, angle="a"),
    sun.DoubleExcitation(0, 2, 1, 3, angle=1.0),
    sun.DoubleExcitation(0, 2, 1, 5, angle="c"),
    sun.DoubleExcitation(0, 2, 5, 3, angle=1.),
    sun.DoubleExcitation(0, 2, 4, 6, angle="e"),
    sun.DoubleExcitation(0, 2, 4, 9, angle=1.),
])
circuit.export_to(filename="unpaired_example.qpic")
circuit.export_to(filename="unpaired_example.pdf")