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

def Braket(backend:str='tequila',*args,**kwargs)->TequilaBraket:
    '''
    Interface for Fermionic Braket <U_bra|H|U_ket>. Since it allows U1 != U2, and U1=U2, some keywords can be presented as
    keywordX -> X \in [_bra,_ket,None] (i.e. init_state,init_state_ket,init_state_bra). If X=None, bra and ket are setted. 
    If X=ket, but not X=bra provided, bra is setted to ket.
    Parameters
    ----------
    backend
        fermionic backend to compute the Braket. See sun.show_supported_modules() or show_available_modules().
    mol/molecule
        tequila or pyscf molecule to get the molecular info and the electronic integrals
    integral_mager + parameters:
        alternative to molecule, tequila objects providing required molecular information
    h/int1e/one_body_integrals + g/int2e/two_body_integrals + (Optional)c/e_core/constant_term  + mo_coeff/orbital_coefficients
    + (Optional) s/ovlp/overlap_integrals + n_elec
        electronic integral information
    (Optional)n_qubits
        number of circuit's qubits. If not provided, assumed from molecular information
    init_stateX: type = [tq.QCircuit,tq.QubitWaveFunction,tc.Circuit(for backend=tcc),ci_vector(list,array),
    any way of initializing a tq.WaveFunction]
        initial state before applying excitations
    reference:
        alternative to init_state
    (Optional)init_stateX_variables 
        initial state varibales, may be provided if init_stateX = tq.QCircuit
    variablesX
        circuit variables
    init_guessX
        variables initital guess for minimization
    bra/ket/circuit/indices: list
        fermionic excitation indices, expected in [[(0,1),(0+n_mo,1+n_mo),...],[(a,b),(c,d),...],...] Format (Upthendown order)
    (Optional, just for backend='tcc') tcc_kwargs
        engine
            see tcc -> UCC engine
        dtype
            see tcc.set_dtype
        backend
            see tcc.set_backend
        else: kwargs provided to the UCC object initialization
    '''
    if 'mol' in kwargs:
        kwargs['molecule'] = kwargs['mol']
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
    return INSTALLED_FERMIONIC_BACKENDS[backend.lower()](*args,**kwargs) 


def from_Qcircuit(circuit:QCircuit,variables:Variables|None=None)->Union[list,QCircuit,list]:
    indices:list = []
    reference:QCircuit = QCircuit()
    params:list=[]
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
                params.append(gate.parameter)
            else:
                temp = []
                params.append(gate.parameter)
                for i in range(len(gate._target)//2):
                    temp.append((gate._target[2*i],gate._target[2*i+1]))
        else:
            raise TequilaException(f'Gate {gate._name}({gate._parameter}) not allowed')
    return indices,reference,params