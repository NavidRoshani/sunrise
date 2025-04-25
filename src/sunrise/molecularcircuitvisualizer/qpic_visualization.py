import subprocess

from .quantum_chemistry import PairCorrelatorGate
from .quantum_chemistry import SingleExcitation
from .generic import Circuit
from .generic import GenericGate
from .quantum_chemistry import DoubleExcitation
from .quantum_chemistry import OrbitalRotatorGate


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
                elif gate != circuit[-1] and circuit[i + 1]._name == 'FermionicExcitation' and len(circuit[i + 1].indices[0]) == 2 and circuit[i + 1].indices[0][0] // 2 == index[0] // 2 and circuit[i + 1].indices[0][1] // 2 == index[1] // 2 and not n_qubits_is_double:  # hope you enjoy this conditional
                    res.append(OrbitalRotatorGate(index[0] // 2, index[1] // 2, angle=gate._parameter, *args, **kwargs))
                    was_ur = True
                else:
                    res.append(SingleExcitation(index[0], index[1], angle=gate._parameter,n_qubits_is_double=n_qubits_is_double, *args, **kwargs))
            else:
                if index[0] // 2 == index[2] // 2 and index[1] // 2 == index[3] // 2 and not n_qubits_is_double:  ## TODO: Maybe generalized for further excitations
                    res.append(PairCorrelatorGate(index[0] // 2, index[1] // 2, angle=gate._parameter, *args, **kwargs))
                else:
                    res.append(DoubleExcitation(index[0], index[1], index[2], index[3], angle=gate._parameter, n_qubits_is_double=n_qubits_is_double, *args,**kwargs))
        elif gate._name == 'QubitExcitation':
            index = gate._target
            if len(index) == 2:
                if was_ur:
                    was_ur = False
                    continue
                elif gate != circuit[-1] and circuit[i + 1]._name == 'QubitExcitation' and len(circuit[i + 1]._target) == 2 and circuit[i + 1]._target[0] // 2 == index[0] // 2 and circuit[i + 1]._target[1] // 2 == index[1] // 2 and not n_qubits_is_double:  # hope you enjoy this conditional
                    res.append(OrbitalRotatorGate(index[0] // 2, index[1] // 2, angle=gate._parameter, *args, **kwargs))
                    was_ur = False
                else:
                    res.append(SingleExcitation(index[0], index[1], angle=gate._parameter,n_qubits_is_double=n_qubits_is_double, *args, **kwargs))
            else:
                if index[0] // 2 == index[2] // 2 and index[1] // 2 == index[3] // 2 and not n_qubits_is_double:  ## TODO: Maybe generalized for further excitations
                    res.append(PairCorrelatorGate(index[0] // 2, index[1] // 2, angle=gate._parameter,*args, **kwargs))
                else:
                    res.append(DoubleExcitation(index[0], index[1], index[2], index[3], angle=gate._parameter, n_qubits_is_double=n_qubits_is_double, *args,**kwargs))
        else:
            res.append(GenericGate(U=gate, name="simple", n_qubits_is_double=n_qubits_is_double, *args, **kwargs))
    return Circuit(res)