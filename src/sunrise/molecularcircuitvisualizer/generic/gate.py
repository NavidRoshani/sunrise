import copy
from typing import List, Optional

import tequila as tq

from ..core import CircuitState, CircuitStyle
from ..core import Gate


# TODO: add documentation, since I did not create this, I am unsure what it actually does and more importantly why
class GenericGate(Gate):
    """
    This is a generic gate, which renders the gate according to its name and QCircuit
    Trotter gates are left just as U.export_to(name.qpic), take care of the qubits yourself
    """


    # TODO: maybe something can be done with the Trotter gates?
    def __init__(self, U: tq.QCircuit, name: Optional[str] =None, n_qubits_is_double: bool = False, daggered: bool = False):
        self.daggered = daggered
        self.U = U
        name_opt = ["initialstate", "simple", "single", "double", "trotter"]
        if name.lower() in name_opt:
            self.name = name.lower()
        else:
            self.name = "simple"
            
        self.qubits = []
        if name == "single" or name == "double":
            for index in U.indices:
                for i in index:
                    self.qubits.append(i)
        elif name == 'trotter':
            self.qubits.extend(U._target)
            self.qubits = list(set(self.qubits))
        else:
            if isinstance(U, tq.QCircuit):
                for gate in self.U.gates:
                    self.qubits.extend(gate.qubits)
                self.qubits = list(set(self.qubits))  # remove duplicates
            elif isinstance(U, tq.circuit._gates_impl.QGateImpl):
                self.qubits.extend(U.qubits)
                self.qubits = list(set(self.qubits))  # remove duplicates
        if n_qubits_is_double:
            spatial = [q // 2 for q in self.qubits]  # half the number of qubits visualized
            self.qubits = list(set(spatial))

    def construct_circuit(self):
        return self.U

    # TODO: this was copied, but can't be right
    def map_variables(self, variables):
        mapped_gate = copy.deepcopy(self)
        for g in mapped_gate.construct_circuit().gates:
            g = g.map_variables(variables)
        return mapped_gate
    
    def used_wires(self) -> List[int]:
        return self.qubits

    def dagger(self) -> "Gate":
        daggered = copy.deepcopy(self)
        daggered.U = daggered.U.dagger()
        daggered.daggered = not daggered.daggered
        return daggered
        

    def _render(self, state: CircuitState, style: CircuitStyle) -> str:
        result = ""
        if self.name != 'trotter':
            for q in self.qubits:
                result += " a{qubit} ".format(qubit=q)
        if self.name == "initialstate":
            result += " G I "
        if self.name == "simple":
            result += " G G "
        if self.name == "single":
            result += " color=blue "
        if self.name == "double":
            result += " color=fai "
        if self.name == "trotter":
            text = tq.circuit.qpic.export_to_qpic(
                circuit=tq.gates.Trotterized(generator=self.U.generator, angle=self.U._parameter))
            while text[0:len("COLOR")] == "COLOR":
                text = text[text.find("\n") + 1:]
            while text[0] == "a":
                text = text[text.find("\n") + 1:]
            result += text

        if self.daggered:
            result += " { $\\dagger$ } "

        return result