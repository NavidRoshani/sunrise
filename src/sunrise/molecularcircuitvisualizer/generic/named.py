import copy
from typing import List, Optional

from ..core import Gate
import tequila as tq


class NamedGate(Gate):
    """
    Simple rectangular gate, with auto-sized width (if not explicitly passed)
    """

    def __init__(self, name: str, circuit: tq.QCircuit, wires: List[int], width: Optional[int] = None):
        self.name = name
        self.circuit = circuit
        self.width = width
        self.wires = wires

    def _render(self, state, style) -> str:
        gates = ""
        for gate in self.wires:
            gates += "a" + str(gate) + " "

        if self.width is None:
            self.width = 10 + len(self.name) * 3

        return gates + "G:width=" + str(self.width) + " {" + self.name + "}"

    def used_wires(self) -> List[int]:
        return self.wires

    def construct_circuit(self) -> tq.QCircuit:
        return self.circuit

    def map_variables(self, variables) -> "NamedGate":
        s = copy.deepcopy(self)
        s.circuit.map_variables(variables)
        return s

    def dagger(self) -> "NamedGate":
        return NamedGate(self.name + "^{\\dagger}", self.circuit.dagger(), self.wires, self.width)
