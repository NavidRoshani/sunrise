import os
from typing import List,Union
from typing_extensions import deprecated

import tequila as tq
import subprocess

from .core import RGB, Colors, Color, ColorRange
from .core import Gate
from .core import CircuitStyle
from .quantum_chemistry import PairCorrelatorGate
from .quantum_chemistry import SingleExcitation
from .generic import Circuit
from .generic import GenericGate
from .quantum_chemistry import DoubleExcitation
from .quantum_chemistry import OrbitalRotatorGate


@deprecated("Instead use the export_qpic method on gate/circuit directly!")
def export_to_qpic(list_of_gates: Union[List[Gate], Gate], filename=None, filepath=None,
                   group_together=False, qubit_names: dict[int, str] = {}, mark_parametrized_gates=False,
                   color_range: bool = False,
                   gatecolor1="tq",
                   textcolor1="white", gatecolor2="fai", textcolor2="white", gatecolor3="unia", textcolor3="black",
                   color_from='blue', color_to='red', **kwargs) -> str:
    """
    This function takes a list of gates (or a single gate) and converts them into a Circuit which is then rendered with the given parameters.

    This function was superseded by the generic Gate::export_qpic() function which has a nicer API.
    It remains only to ensure backwards compatibility with legacy code by converting these parameters into the new structure.
    """

    # define colors as list of dictionaries with "name":str, "rgb":tuple entries
    custom_colors: dict[str, RGB] = {}
    if "colors" in kwargs:
        colors = kwargs["colors"]
        kwargs.pop("colors")

        for color in colors:
            custom_colors[color["name"]] = RGB(*tuple(color["rgb"]))

    wcolors = {}
    if "wire_colors" in kwargs:
        for index, name in kwargs["wire_colors"].items():
            wcolors[index] = Color(name)
        kwargs.pop("wire_colors")

    marking = None
    if mark_parametrized_gates:
        marking = Colors(Color(textcolor3), Color(gatecolor3))

    colored_range = None
    if color_range:
        colored_range = ColorRange(Color(color_from), Color(color_to))

    if isinstance(list_of_gates, List):
        circuit = Circuit(list_of_gates)
    else:
        circuit = list_of_gates

    style = CircuitStyle(group_together, marking, colored_range, Colors(Color(textcolor1), Color(gatecolor1)),
                         Colors(Color(textcolor2), Color(gatecolor2)))

    return circuit.export_qpic(filename, filepath, style, qubit_names, custom_colors, wcolors)


def qpic_to_pdf(filename, filepath=None):
    """Calls qpic to generate a pdf from the given apic file."""

    if filepath:
        subprocess.call(["qpic", "{}.qpic".format(filename), "-f", "pdf"], cwd=filepath)
    else:
        subprocess.call(["qpic", "{}.qpic".format(filename), "-f", "pdf"])


def qpic_to_png(filename, filepath=None):
    """Calls qpic to generate a png from the given apic file."""

    if filepath:
        subprocess.call(["qpic", "{}.qpic".format(filename), "-f", "png"], cwd=filepath)
    else:
        subprocess.call(["qpic", "{}.qpic".format(filename), "-f", "png"])


def from_circuit(U, n_qubits_is_double: bool = False, *args, **kwargs) -> Circuit:
    """
    Converts a tequila QCircuit into a Circuit object, which is renderable with `export_qpic`.
    """

    circuit = U._gates
    res = []
    was_ur = False
    for i, gate in enumerate(circuit):
        if gate._name == 'FermionicExcitation':
            index = ()
            for pair in gate.indices:
                for so in pair:
                    index += (so,)
            if len(index) == 2:
                if was_ur:
                    was_ur = False
                    continue
                elif gate != circuit[-1] and circuit[i + 1]._name == 'FermionicExcitation' and len(
                        circuit[i + 1].indices[0]) == 2 and circuit[i + 1].indices[0][0] // 2 == index[0] // 2 and \
                        circuit[i + 1].indices[0][1] // 2 == index[1] // 2:  # hope you enjoy this conditional
                    res.append(OrbitalRotatorGate(index[0] // 2, index[1] // 2, angle=gate._parameter, *args, **kwargs))
                    was_ur = True
                else:
                    res.append(SingleExcitation(index[0], index[1], angle=gate._parameter, *args, **kwargs))
            else:
                if index[0] // 2 == index[2] // 2 and index[1] // 2 == index[
                    3] // 2:  ## TODO: Maybe generalized for further excitations
                    res.append(PairCorrelatorGate(index[0] // 2, index[1] // 2, angle=gate._parameter, *args, **kwargs))
                else:
                    res.append(DoubleExcitation(index[0], index[1], index[2], index[3], angle=gate._parameter, *args,
                                                **kwargs))
        elif gate._name == 'QubitExcitation':
            index = gate._target
            if len(index) == 2:
                if was_ur:
                    was_ur = False
                    continue
                elif gate != circuit[-1] and circuit[i + 1]._name == 'QubitExcitation' and len(
                        circuit[i + 1]._target) == 2 and circuit[i + 1]._target[0] // 2 == index[0] // 2 and \
                        circuit[i + 1]._target[1] // 2 == index[1] // 2:  # hope you enjoy this conditional
                    res.append(OrbitalRotatorGate(index[0] // 2, index[1] // 2, angle=gate._parameter, *args, **kwargs))
                    was_ur = False
                else:
                    res.append(SingleExcitation(index[0], index[1], angle=gate._parameter, *args, **kwargs))
            else:
                if index[0] // 2 == index[2] // 2 and index[1] // 2 == index[
                    3] // 2:  ## TODO: Maybe generalized for further excitations
                    res.append(PairCorrelatorGate(index[0] // 2, index[1] // 2, angle=gate._parameter, *args, **kwargs))
                else:
                    res.append(DoubleExcitation(index[0], index[1], index[2], index[3], angle=gate._parameter, *args,
                                                **kwargs))
        else:
            res.append(GenericGate(U=gate, name="simple", n_qubits_is_double=n_qubits_is_double, *args, **kwargs))
    return Circuit(res)