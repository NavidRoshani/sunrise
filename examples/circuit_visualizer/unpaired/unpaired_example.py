import sunrise as sun

circuit =sun.Circuit([
    sun.SingleExcitation(0, 3, angle="a"),
    sun.DoubleExcitation(0, 2, 1, 3, angle=1.0),
    sun.DoubleExcitation(0, 2, 1, 5, angle="c"),
    sun.DoubleExcitation(0, 2, 5, 3, angle=1.),
    sun.DoubleExcitation(0, 2, 4, 6, angle="e"),
    sun.DoubleExcitation(0, 2, 4, 9, angle=1.),
])
sun.export_qpic(circuit, filename="unpaired_example.qpic")