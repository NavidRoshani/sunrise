import sunrise as sun
from math import pi
from sunrise.graphical import *

circuit =GraphicalCircuit([
    SingleExcitation(0, 2, angle=0.),
    SingleExcitation(0, 2, angle=pi / 6),
    SingleExcitation(0, 2, angle=pi / 4),
    SingleExcitation(0, 2, angle=pi / 2),
    SingleExcitation(0, 2, angle=pi),
    SingleExcitation(0, 2, angle=3 * pi / 4),
    SingleExcitation(0, 2, angle=pi),
    SingleExcitation(0, 2, angle=4 * pi),
    SingleExcitation(1, 3, angle=0., unit_of_pi=True),
    SingleExcitation(1, 3, angle=1 / 6, unit_of_pi=True),
    SingleExcitation(1, 3, angle=1 / 4, unit_of_pi=True),
    SingleExcitation(1, 3, angle=1 / 2, unit_of_pi=True),
    SingleExcitation(1, 3, angle=1, unit_of_pi=True),
    SingleExcitation(1, 3, angle=3 / 4, unit_of_pi=True),
    SingleExcitation(1, 3, angle=1, unit_of_pi=True),
    SingleExcitation(1, 3, angle=4, unit_of_pi=True),
])
circuit.export_qpic(filename="color_range_example")
circuit.export_to("color_range_example.pdf")