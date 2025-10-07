import sunrise as sun
from sunrise.graphical import *

circuit =GraphicalCircuit([
    SingleExcitation(0, 3, angle="a"),
    DoubleExcitation(0, 2, 1, 3, angle=1.0),
    DoubleExcitation(0, 2, 1, 5, angle="c"),
    DoubleExcitation(0, 2, 5, 3, angle=1.),
    DoubleExcitation(0, 2, 4, 6, angle="e"),
    DoubleExcitation(0, 2, 4, 9, angle=1.),
])
circuit.export_to(filename="unpaired_example.qpic")
circuit.export_to(filename="unpaired_example.pdf")