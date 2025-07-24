from tequila import QCircuit,TequilaException
from tequila.objective.objective import Variables
from tequila.circuit.gates import QubitExcitationImpl
from tequila.quantumchemistry.chemistry_tools import FermionicGateImpl
from sunrise.expval.tequila_expval import TequilaBraket
from tequila import TequilaException
from typing import  Union
from numbers import Number

SUPPORTED_FERMIONIC_BACKENDS = ["tequila", 'fqe', "tcc"]
INSTALLED_FERMIONIC_BACKENDS = {}#{"tequila": TequilaBraket}

try:
    from sunrise.expval.tcc_expval import TCCBraket
    INSTALLED_FERMIONIC_BACKENDS["tcc"] = TCCBraket
except ImportError:
    pass
try:
    from sunrise.expval.fqe_expval import FQEBraket

    INSTALLED_FERMIONIC_BACKENDS["fqe"] = FQEBraket
except ImportError:
    pass

def show_available_modules():
    print("Available Fermionic Modules:")
    for k in INSTALLED_FERMIONIC_BACKENDS.keys():
        print(k)


def show_supported_modules():
    print(SUPPORTED_FERMIONIC_BACKENDS)

def Braket(molecule=None,indices:list[list,tuple]|None=None,reference:list[str,list,QCircuit]|None=None,backend:str='tequila',optimizer=None,operator=None,*args,**kwargs)->TequilaBraket:
    if 'mol' in kwargs:
        if molecule is not None:
            raise TequilaException("Two Molecules Provided?")
        else:
            molecule = kwargs['mol']
            kwargs.pop('mol')
    if 'circuit' in kwargs:
        if (U is not None) and (indices is not None):
            raise TequilaException("More than one Circuit Provided?")
        else:
            temp = kwargs['circuit']
            kwargs.pop('circuit')
            if isinstance(temp,QCircuit):
                U = temp
            else:
                indices = temp
    #any kwargs and circuit form should be managed inside each class
    return INSTALLED_FERMIONIC_BACKENDS[backend.lower()](molecule=molecule,reference=reference,indices=indices,optimizer=optimizer,operator=operator,*args,**kwargs) #TODO: DO something not anoying


def from_Qcircuit(circuit:QCircuit,variables:Variables|None=None)->Union[list,QCircuit]:
    indices:list = []
    reference:QCircuit = QCircuit()
    begining = True
    if variables is not None:
        circuit = circuit.map_variables(variables)
    for gate in circuit.gates:
        if begining and not hasattr(gate,'_parameter') or isinstance(gate._parameter,Number):
            reference += gate
        elif isinstance(gate,QubitExcitationImpl): #maybe we can consider other gates but basic implementation for the moment
            if isinstance(gate._parameter,Number):
                reference += gate
            elif isinstance(gate,FermionicGateImpl):
                begining = False
                indices.append(gate.indices)
            else:
                temp = []
                for i in range(len(gate._target)//2):
                    temp.append((gate._target[2*i],gate._target[2*i+1]))
        else:
            raise TequilaException(f'Gate {gate._name}({gate._parameter}) not allowed')
    return indices,reference