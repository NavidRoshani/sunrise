from typing import List
from copy import deepcopy
from ..core import Gate
import tequila as tq


class Circuit(Gate):
    """
    Basic wrapper class for multiple gates.
    Gates can be added in a similar as with tequila's QCircuit: circuit += gate
    """

    gates: List[Gate]

    def __init__(self, gates: List[Gate] = []):
        if gates is not None: self.gates = gates

    def __add__(self, other: Gate):
        self.gates.append(other)
        return self

    def dagger(self) -> "Circuit":
        return Circuit([gate.dagger() for gate in reversed(self.gates)])

    def _render(self, state, style) -> str:
        output = ""
        sim = self._simplify_circuit()
        for gate in sim.gates:
            output += gate.render(state, style) + " \n"
        return output

    def used_wires(self) -> List[int]:
        wires = set()
        for gate in self.gates:
            wires.update(gate.used_wires())
        return list(wires)

    def construct_circuit(self) -> tq.QCircuit:
        U = tq.QCircuit()
        for gate in self.gates:
            U += gate.construct_circuit()
        return U

    def map_variables(self, variables) -> "Circuit":
        gates = []
        for gate in self.gates:
            gates.append(gate.map_variables(variables))
        return Circuit(gates)
    def _simplify_circuit(self) -> "Circuit":
        from ..generic import GenericGate
        cp = deepcopy(self)
        for g in range(len(self.gates)-1)[::-1]:
            if isinstance(self.gates[g],GenericGate) and isinstance(self.gates[g+1],GenericGate):
                cp.gates[g].U += cp.gates[g+1].U
                cp.gates[g].qubits = [*set(cp.gates[g].qubits+cp.gates[g+1].qubits)]
                cp.gates.pop(g+1)
        return cp