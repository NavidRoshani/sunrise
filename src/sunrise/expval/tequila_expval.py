import tequila as tq
from tequila import BraKet,QCircuit,QubitHamiltonian,QubitWaveFunction,BitString
from numbers import Number
from tequila import TequilaException
from numpy import ndarray

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

    def energy(self,variables:dict={}):
        pass

    def _from_indices(indices:list=None)->QCircuit:
        pass