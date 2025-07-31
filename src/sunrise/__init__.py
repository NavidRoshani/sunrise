from sunrise.hybrid_base import Molecule
from sunrise.plot_MO import plot_MO
from sunrise.hybridization.hybridization import Graph
from sunrise.molecularcircuitvisualizer.qpic_visualization import qpic_to_pdf,qpic_to_png,from_circuit
from sunrise.molecularcircuitvisualizer.generic import *
from sunrise.molecularcircuitvisualizer.quantum_chemistry import *
from sunrise.hcb_measurement.measurement_utils import *
from sunrise.fermionic_excitation.orb_rotation import OrbitalRotation
from sunrise.miscelaneus.giusepe import giuseppe
from sunrise.miscelaneus.bar import giussepe_bar
from sunrise.expval import Braket,show_available_modules,show_supported_modules

import tequila as tq
class FermionicCircuit:

    def __init__(self, operations=None, initial_state=None):
        self._operations = operations
        if operations is None:
            self._operations = []
        self._initial_state = initial_state

        self.verify()

    def __add__(self, other):
        initial_state = self._initial_state
        operations = self._operations + other._operations
        if hasattr(other, "_operations"):
            if self._initial_state is None:
                initial_state = other._initial_state
            elif other._initial_state != self._initial_state:
                raise Exception(f"FermionicCircuit + FermionicCircuit with two different initial states:\n{self._initial_state}, {other._initial_state}")

        return FermionicCircuit(operations=operations, initial_state=initial_state)

    def verify(self):
        operations = [[tq.assign_variable(x[0]), tuple(x[1])] for x in self._operations]
        self._operations = operations

    def extract_variables(self):
        pass

    @property
    def variables(self):
        return self.extract_variables()

def make_excitation_gate(indices, angle, *args, **kwargs):
    return (angle, indices)

