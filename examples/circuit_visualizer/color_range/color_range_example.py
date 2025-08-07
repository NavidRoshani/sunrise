import sunrise as sun
from math import pi

circuit =sun.GraphicalCircuit([
    sun.SingleExcitation(0, 2, angle=0.),
    sun.SingleExcitation(0, 2, angle=pi / 6),
    sun.SingleExcitation(0, 2, angle=pi / 4),
    sun.SingleExcitation(0, 2, angle=pi / 2),
    sun.SingleExcitation(0, 2, angle=pi),
    sun.SingleExcitation(0, 2, angle=3 * pi / 4),
    sun.SingleExcitation(0, 2, angle=pi),
    sun.SingleExcitation(0, 2, angle=4 * pi),
    sun.SingleExcitation(1, 3, angle=0., unit_of_pi=True),
    sun.SingleExcitation(1, 3, angle=1 / 6, unit_of_pi=True),
    sun.SingleExcitation(1, 3, angle=1 / 4, unit_of_pi=True),
    sun.SingleExcitation(1, 3, angle=1 / 2, unit_of_pi=True),
    sun.SingleExcitation(1, 3, angle=1, unit_of_pi=True),
    sun.SingleExcitation(1, 3, angle=3 / 4, unit_of_pi=True),
    sun.SingleExcitation(1, 3, angle=1, unit_of_pi=True),
    sun.SingleExcitation(1, 3, angle=4, unit_of_pi=True),
])
circuit.export_qpic(filename="color_range_example")
circuit.export_to("color_range_example.pdf")