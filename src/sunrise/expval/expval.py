from tequila import QCircuit
from .tequila_expval import TequilaBraket
from tequila import TequilaException


SUPPORTED_FERMIONIC_BACKENDS = ["tequila", 'fqe', "tcc"]
INSTALLED_FERMIONIC_BACKENDS = {"tequila": TequilaBraket}

try:
    from .tcc_expval import TCCBraket

    INSTALLED_FERMIONIC_BACKENDS["tcc"] = TCCBraket
except ImportError:
    pass
try:
    from .fqe_expval import FQEBraket

    INSTALLED_FERMIONIC_BACKENDS["fqe"] = FQEBraket
except ImportError:
    pass

def show_available_modules():
    print("Available Fermionic Modules:")
    for k in INSTALLED_FERMIONIC_BACKENDS.keys():
        print(k)


def show_supported_modules():
    print(SUPPORTED_FERMIONIC_BACKENDS)

def Braket(molecule=None,U:QCircuit=None,indices:[list,tuple]=None,reference:[str,list,QCircuit]=None,backend:str='tequila',optimizer=None,operator=None,*args,**kwargs)->TequilaBraket:
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
    return INSTALLED_FERMIONIC_BACKENDS[backend.lower()](molecule=molecule,U=U,reference=reference,indices=indices,optimizer=optimizer,operator=operator,*args,**kwargs)
