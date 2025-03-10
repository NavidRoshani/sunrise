import abc
from os import path
from typing import Optional, List

import tequila as tq

from .color import RGB, DefaultColors, Color
from .state import CircuitState, CircuitStyle


class Gate(abc.ABC):
    """
    Gate that can generate it's qpic visualization on its own as well as construct the matching tequila circuit.
    """

    @abc.abstractmethod
    def construct_circuit(self) -> tq.QCircuit:
        """Constructs the matching QCircuit"""
        pass

    @abc.abstractmethod
    def map_variables(self, variables) -> "Gate":
        pass


    def render(self, state: CircuitState, style: CircuitStyle) -> str:
        """
        This method renders the gate to a qpic string.
        This is the generic wrapper adding styling which is the same for each gate.
        When rendering subgates ALWAYS use this method.
        """
        out = self._render(state, style)
        if style.group_together:
            out += "\n" + state.all_names() + " TOUCH\n"
        return out


    @abc.abstractmethod
    def _render(self, state: CircuitState, style: CircuitStyle) -> str:
        """
        This method should return the qpic string for this gate, tailing \n is not needed!
        Do NOT call this method directly, instead use the generic render() method!
        """
        pass


    def export_qpic(self, filename: str, filepath: Optional[str] = None, style: CircuitStyle = CircuitStyle(), labels: dict[int, str] = {}, colors: dict[str, RGB] = {}, wire_colors: dict[int, Color] = {} , select:dict=None):
        """
        This method renders this gate/circuit into a complete qpic document and stores it in the given filepath (or cwd if omitted).
        Parameters:
            filename -- Name of the resulting file
            filepath -- If given, the qpic file will be stored in this directory.
            style -- Defines the styling that should be used to render the circuit.
            labels -- Dictionary mapping wire indices to names. These well be displayed instead of the index.
            colors -- Dictionary of additional colors that should be defined. The DefaultColors are prepended and can thus be overridden.
            wire_colors -- Dictionary mapping wire indices to a Color. If set the wire will be colored instead of defaulting to black.
        """
        result = ""

        if not len(wire_colors) and select is not None:
            for s in select:
                if select[s] == 'B':
                    wire_colors[s] = "red"
        # add colors
        for color, rgb in DefaultColors.items():
            result += f"COLOR {color} {rgb.r} {rgb.g} {rgb.b}\n"

        for color, rgb in colors.items():
            result += f"COLOR {color} {rgb.r} {rgb.g} {rgb.b}\n"

        # add wires
        wires = self.used_wires()
        for wire in range(0, max(wires) + 1):
            name = "a" + str(wire)

            if wire in labels:
                label = labels[wire]
            else:
                label = str(wire)

            if wire in wire_colors:
                color = wire_colors[wire]
            else:
                color = "black"
            result += f"color={color} {name} W {label} \n"

        for wire in range(0, max(wires) + 1):
            result += f"a{wire} /\n"

        # add gates
        state = CircuitState(max(wires))
        result += self.render(state, style) + "\n"

        if filename is not None:
            extendedFilename = filename
            if not extendedFilename.endswith(".qpic"):
                extendedFilename = filename + ".qpic"
            if filepath is not None:
                extendedFilename = path.join(filepath, extendedFilename)
            with open(extendedFilename, "w") as file:
                file.write(result)

    @abc.abstractmethod
    def used_wires(self) -> List[int]:
        """
        This method should return all qubits that are used by this gate.
        """
        pass

    @abc.abstractmethod
    def dagger(self) -> "Gate":
        """
        This method should return the daggered gate.
        """
        pass
