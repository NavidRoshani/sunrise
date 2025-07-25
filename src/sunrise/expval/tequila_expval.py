import tequila as tq
from tequila import BraKet,QCircuit,QubitHamiltonian,QubitWaveFunction,BitString
from numbers import Number
from tequila import TequilaException
from numpy import ndarray
from tequila import QCircuit,TequilaException,Molecule,QubitWaveFunction,simulate,Variable,Objective
from tequila.objective.objective import Variables
from tequila.quantumchemistry import qc_base
from tequila.utils.bitstrings import BitString
from numbers import Number
from numpy import array,ceil,pi
from pyscf.gto import Mole
from sunrise.expval.pyscf_molecule import from_tequila
from copy import deepcopy
from typing import Union

class TequilaBraket():
    def __init__(self,U:list[QCircuit,list,tuple]=None,bra:QCircuit=None,ket:QCircuit=None,reference:QCircuit=None,operator:QubitHamiltonian=None,*args,**kwargs):
        '''
        #TODO: Define all posible inputs
        '''
        if 'circuit' in kwargs:
            circuit = kwargs['circuit']
            kwargs.pop('circuit')
            if U is not None:
                raise TequilaException('Two circuits provided?')
            else: U = circuit
        #TODO Here he should check also bra and ket before
        if U is not None and isinstance(U,(list,tuple)):
            if isinstance(U[0],Number):
                U = [[U,],]
            elif isinstance(U[0][0],Number):
                U = [U,]
            if "molecule" in kwargs:
                molecule = kwargs['molecule']
            else:
                if 'transformation' in kwargs:
                    trans = kwargs['transformation']
                    kwargs.pop('transformation')
                elif 'upthendown' in kwargs:
                    trans = 'reversedjordanwigner' if kwargs['upthendown']  else 'jordanwigner' 
                    kwargs.pop['upthendown']
                elif 'reverse' in kwargs:
                    trans = 'reversedjordanwigner' if kwargs['reverse']  else 'jordanwigner' 
                    kwargs.pop['reverse']
                else: 
                    trans = 'jordanwigner'

                nsos = max([max([max([idx[0] for idx in exct]+[idx[1] for idx in exct])]) for exct in U])
                geom = ''
                for i in range(nsos//2+1):
                    geom += f'H 0. 0. {i}'
                molecule = tq.Molecule(geometry=geom,basis_set='sto-3g',transformation=trans)
            Ux = QCircuit()
            for exct in U:
                Ux+= molecule.make_excitation_gate(indices=exct,angle=str(exct))
            U = Ux
        else:
            molecule = None
        if U is not None and ket is not None:
            raise TequilaException('Two circuit provided?')
        elif U is not None:
            ket = U
        if reference is not None:
            if isinstance(reference,QCircuit):
                if 'init_vars' in kwargs: #UGLY: Better keyword name?
                    ivar = kwargs['init_vars']
                    kwargs.pop('init_vars')
                    reference = reference.map_variables(ivar)                    
                ket = reference + ket #OPTIMIZE: Maybe cheaper to set the reference as initial state?
                if bra is not None:
                    bra = reference + bra
                else:
                    bra = reference
                self.reference = None
            elif reference is None or isinstance(reference,QubitWaveFunction):
                self.reference = reference
            else:
                #Reference expected to be whatever info to initialize a QubitWaveFunction
                if isinstance(reference,(int, BitString)):
                    if 'n_qubits' in kwargs:
                        n_qubits = kwargs['n_qubits']
                        kwargs.pop('n_qubits')
                    else:
                        raise TequilaException('')
                    reference = QubitWaveFunction.from_basis_state(n_qubits=n_qubits,basis_state=reference)                    
                else:
                    try:
                        reference = QubitWaveFunction.convert_from(val=reference)
                    except:
                        raise TequilaException(f'Refernce state format not recognized, provided {type(reference).__name__}')
        if 'H' in kwargs:
            H = kwargs['H']
            kwargs.pop('H')
            if operator is not None:
                raise TequilaException('Two Operators provided?')
            else:
                operator = H
        self.operator = operator
        self.molecule = molecule #actually dont sure if need to save, only if we add the option to update the circuit once created
        self.variables:dict = None
        self.opt_res = None
        self.ket = ket
        self.bra = bra

    def minimize(self,**kwargs)->float:
        pass

    def simulate(self,params:Union[list,dict])->float:
        pass

    @property
    def energy(self)->float:
        pass

    @property
    def bra(self):
        """
        Excitation operators applied to the bra.
        """
        pass
    
    @bra.setter
    def bra(self, bra):
        '''
        Expected indices in [[(0,2),(1,3),...],[(a,b),(c,d),...],...] Format (Upthendown order)
        '''
        pass
    
    @property
    def ket(self):
        """
        Excitation operators applied to the ket.
        """
        pass
    
    @ket.setter
    def ket(self, ket) -> None:
        '''
        Expected indices in [[(0,2),(1,3),...],[(a,b),(c,d),...],...] Format (Upthendown order)
        '''
        pass

    @property
    def variables_bra(self) -> dict:
        """Tequila Circuit Bra variables."""
        pass
    
    @property
    def params_bra(self):
        """TCC Circuit Bra parameters (values after minimization or variables name)."""
        pass
    
    @variables_bra.setter
    def variables_bra(self, variables_bra):
        '''
        See TCC variables
        '''
        pass

    @property
    def variables_ket(self):
        """Tequila circuit Ket parameters."""
        pass
    
    @property
    def params_ket(self):
        """TCC Circuit Ket parameters (values after minimization or variables name)."""
        pass
    
    @variables_ket.setter
    def variables_ket(self, variables_ket):
        '''
        See TCC variables
        '''
        pass
    
    @property
    def variables(self) -> dict:
        """Tequila circuit variables."""
        pass 
    
    @property
    def params(self):
        """TCC parameters (values after minimization or variables name)."""
        pass
    
    @variables.setter
    def variables(self, variables):
        """Tequila circuit variables."""
        pass

    @property
    def init_state_bra(self):
        """
        The circuit initial state before applying the excitation operators. Usually RHF.
        """
        pass
    
    @property
    def init_state_ket(self):
        """
        The circuit initial state before applying the excitation operators. Usually RHF.
        """
        pass
    
    @property
    def init_state(self):
        """
        The circuit initial state before applying the excitation operators. Usually RHF.
        """
        pass

    @init_state_bra.setter
    def init_state_bra(self, init_state_bra):
        pass
    
    @init_state_ket.setter
    def init_state_ket(self, init_state_ket):
        pass

    @init_state.setter
    def init_state(self, init_state):
        pass

    @property
    def init_guess(self):
        """
        Initial Angle Value for minimization, default all 0.
        """
        pass
    
    @init_guess.setter
    def init_guess(self, init_guess):
        pass

def init_state_from_wavefunction(wvf:QubitWaveFunction):
    if not isinstance(wvf._state,dict):
        return init_state_from_array(wvf=wvf)
    init_state = []
    for i in wvf._state:
        vec = bin(i)[2:]
        if len(vec) < wvf.n_qubits:
            vec = '0'*(wvf.n_qubits-len(vec))+vec
        init_state.append([vec[::-1],wvf._state[i].real])#tcc automatically does this, but with an anoying message everytime
    return init_state

def init_state_from_array(wvf:QubitWaveFunction,tol=1e-6):
    '''
    Expected Initial State in UptheDown
    '''
    if isinstance(wvf._state,dict):
        return init_state_from_wavefunction(wvf)
    init_state = []
    nq = wvf.n_qubits
    nq = int(2* (ceil(nq/2))) #if HCB may be odd amount of qubits
    for i,idx in enumerate(wvf._state):
        vec = BitString.from_int(i)
        vec.nbits = nq
        vec = vec.binary
        if abs(idx) > tol:
            init_state.append([vec[::-1],idx.real]) #tcc automatically does this, but with an anoying message everytime
    return init_state

def from_indices(indices):
    '''
    Expected indices like [[(0,1),(n_mo+0,n_mo+1),...],[(a,b),(c,d),...],...] (in upthendown)
    Returned [(0,n_no+0,...,n_mo+1,1),(a,c,...,d,b)]
    '''
    if indices is None:
        return None,None,None
    assert isinstance(indices,(list,tuple))
    if isinstance(indices[0],Number):
        indices = [indices]
    ex_ops = []
    params = []
    param_ids = []
    for exct in indices:
        exc = []
        params.append(str(exct))
        param_ids.append(len(param_ids))
        for idx in exct:
            exc.append(idx[0])
            exc.insert(0,idx[1])
        ex_ops.append(tuple(exc))
    return ex_ops,params,param_ids

def map_variables(x:list[Variable,Objective],dvariables:dict):
    if isinstance(x,Variable):
        x = x.map_variables(dvariables)
    elif isinstance(x,Objective):
        x=simulate(x,dvariables)
    return x