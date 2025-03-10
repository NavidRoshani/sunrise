from src.sunrise.molecularcircuitvisualizer import qpic_visualization as qp
from math import pi

circuit =[
    qp.SingleExcitation(0, 2, angle=0.),
    qp.SingleExcitation(0, 2, angle=pi / 6),
    qp.SingleExcitation(0, 2, angle=pi / 4),
    qp.SingleExcitation(0, 2, angle=pi / 2),
    qp.SingleExcitation(0, 2, angle=pi),
    qp.SingleExcitation(0, 2, angle=3 * pi / 4),
    qp.SingleExcitation(0, 2, angle=pi),
    qp.SingleExcitation(0, 2, angle=4 * pi),
    qp.SingleExcitation(1, 3, angle=0., unit_of_pi=True),
    qp.SingleExcitation(1, 3, angle=1 / 6, unit_of_pi=True),
    qp.SingleExcitation(1, 3, angle=1 / 4, unit_of_pi=True),
    qp.SingleExcitation(1, 3, angle=1 / 2, unit_of_pi=True),
    qp.SingleExcitation(1, 3, angle=1, unit_of_pi=True),
    qp.SingleExcitation(1, 3, angle=3 / 4, unit_of_pi=True),
    qp.SingleExcitation(1, 3, angle=1, unit_of_pi=True),
    qp.SingleExcitation(1, 3, angle=4, unit_of_pi=True),
]
qp.export_to_qpic(circuit,filename="color_range_example",color_range=True)