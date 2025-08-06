import tencirchem as tcc
from sunrise.expval.tcc_engine.braket import EXPVAL
from ..fermionic_excitation.circuit import FCircuit
from tequila import TequilaException,Molecule,QubitWaveFunction,simulate,Variable,Objective
from tequila.objective.objective import Variables
from tequila.quantumchemistry.chemistry_tools import NBodyTensor
from tequila.quantumchemistry import qc_base
from tequila.utils.bitstrings import BitString
from numbers import Number
from numpy import array,ceil,argwhere
from pyscf.gto import Mole
from sunrise.expval.pyscf_molecule import from_tequila
from copy import deepcopy
from typing import Union

#assumed to be installed pyscf since dependency for sunrise and tcc

class TCCBraket:
    def __init__(self,bra:FCircuit|None=None,ket:FCircuit|None=None,backend_kwargs:dict|None={},*args,**kwargs):

        if 'engine' in backend_kwargs:
            engine = backend_kwargs['engine']
            backend_kwargs.pop('engine')
        else: engine = None
        if 'backend' in  backend_kwargs:
            tcc.set_backend(backend_kwargs['backend'])
            backend_kwargs.pop('backend')
        if 'dtype' in backend_kwargs:
            tcc.set_dtype(backend_kwargs['dtype'])
            backend_kwargs.pop('dtype')
        
        if 'circuit' in kwargs:
            circuit = kwargs['circuit']
            kwargs.pop('circuit')
            if ket is not None:
                raise TequilaException('Two circuits provided?')
            else:
                ket = circuit

        run_hf = (bra is None or bra.initial_state is None) and (ket is None or ket.initial_state is None)   
        if 'molecule' in kwargs and kwargs['molecule']:
            molecule = kwargs['molecule']
            kwargs.pop('molecule')
            if isinstance(molecule,qc_base.QuantumChemistryBase):
                aslst = [i.idx_total for i in molecule.integral_manager.active_orbitals]
                active_space = (molecule.n_electrons,molecule.n_orbitals)
                molecule = from_tequila(molecule)
            elif isinstance(molecule,Mole):
                aslst = [*range(molecule.nao_nr_range)]
                active_space = (molecule.nelectron,molecule.nao_nr)
            self.BK:EXPVAL = EXPVAL(mol=molecule,run_hf= run_hf, run_mp2= False, run_ccsd= False, run_fci= False,init_method="zeros",aslst=aslst,active_space=active_space,engine=engine,**backend_kwargs)
        elif 'integral_manager' in kwargs and 'parameters' in kwargs:
            integral = kwargs['integral_manager']
            params = kwargs['parameters']
            kwargs.pop('integral_manager')
            kwargs.pop('parameters')
            molecule = Molecule(parameters=params,integral_manager=integral)
            aslst = [i.idx_total for i in integral.active_orbitals]
            active_space = (molecule.n_electrons,molecule.n_orbitals)
            molecule = from_tequila(molecule)
            self.BK:EXPVAL = EXPVAL(mol=molecule,run_hf= run_hf, aslst=aslst,active_space=active_space,run_mp2= False, run_ccsd= False, run_fci= False,init_method="zeros",**backend_kwargs)
        else:
            int1e = None
            int2e = None
            e_core = None
            mo_coeff = None
            ovlp = None
            if "int1e"  in kwargs:
                int1e = kwargs['int1e']
                kwargs.pop('int1e')
            elif "one_body_integrals"  in kwargs:
                int1e = kwargs['one_body_integrals']
                kwargs.pop('one_body_integrals')
            elif "h"  in kwargs:
                int1e = kwargs['h']
                kwargs.pop('h')
            if 'int2e' in kwargs:
                int2e = kwargs['int2e']
                kwargs.pop('int2e')
            elif 'two_body_integrals' in kwargs:
                int2e = kwargs['two_body_integrals']
                kwargs.pop('two_body_integrals')
            elif 'g' in kwargs:
                int2e = kwargs['g']
                kwargs.pop('g')
            if isinstance(int2e,NBodyTensor):
                int2e = int2e.elems
            if 'e_core' in kwargs:
                e_core = kwargs['e_core']
                kwargs.pop('e_core')
            elif 'constant_term' in kwargs:
                e_core = kwargs['constant_term']
                kwargs.pop('constant_term')
            elif 'c' in kwargs:
                e_core = kwargs['c']
                kwargs.pop('c')    
            else: e_core = 0.
            if 'mo_coeff' in kwargs:
                mo_coeff = kwargs['mo_coeff']
                kwargs.pop('mo_coeff')
            elif 'orbital_coefficients' in kwargs:
                mo_coeff = kwargs['orbital_coefficients']
                kwargs.pop('orbital_coefficients')
            if 'ovlp' in kwargs:
                ovlp = kwargs['ovlp']
                kwargs.pop('ovlp')
            elif 'overlap_integrals' in kwargs:
                ovlp = kwargs['overlap_integrals']
                kwargs.pop('overlap_integrals')
            elif 's' in kwargs:
                ovlp = kwargs['s']
                kwargs.pop('s')
            if 'n_elec' in kwargs:
                n_elec=kwargs['n_elec']
                kwargs.pop('n_elec')
            elif 'n_electrons' in kwargs: 
                n_elec=kwargs['n_elec']
                kwargs.pop('n_elec')
            elif ket is not None and ket.init_state is not None:
                if isinstance(ket.initial_state._state,dict):
                    n_elec = bin([*ket.initial_state._state.keys()][0])[2:].count('1')
                else:
                    n_elec = bin(argwhere(ket.init_state._state>1.e-6)[0][0])[2:].count('1')
            else:
                raise TequilaException("No manner of defining the amount of electrons provided")
            if all([i is not None for i in[int2e,int1e,mo_coeff]]):
                self.BK:EXPVAL = EXPVAL.from_integral(int1e=int1e, int2e=int2e,n_elec= n_elec, e_core=e_core,ovlp=ovlp,mo_coeff=mo_coeff,init_method="zeros",run_hf= run_hf, run_mp2= False, run_ccsd= False, run_fci= False,**backend_kwargs)
            else:
                raise TequilaException('Not enough molecular data provided')

        if bra is not None and bra.initial_state is not None: #init state initialization splitted bcs we need BK structe, but we get info to initialize BK from init state
            ini_bra = init_state_from_wavefunction(bra.initial_state)
            for i in ini_bra:
                i[0] = self.BK.get_addr(i[0])
            init = [0] * self.BK.civector_size
            for i in ini_bra:
                init[i[0]] = i[1]
            self.init_state_bra = init
        if ket is not None and ket.initial_state is not None: #TCC builds HF if None provided
            ini_ket = init_state_from_wavefunction(ket.initial_state)
            for i in ini_ket:
                i[0] = self.BK.get_addr(i[0])
            init = [0] * self.BK.civector_size
            for i in ini_ket:
                init[i[0]] = i[1]
            self.init_state_ket = init
        


        if ket is not None and ket.variables is not None:
            self.variables_ket = ket.variables
        if bra is not None and bra.variables is not None:
            self.variables_bra = bra.variables

        if ket is not None:
            ket = ket.to_upthendown(len(self.BK.aslst))
            self.ket = ket.extract_indices()
        if bra is not None:
            bra = bra.to_upthendown(len(self.BK.aslst))
            self.bra = bra.extract_indices() 
        self.opt_res = None
        
    def minimize(self,**kwargs)->float:
        if 'init_guess_bra' in kwargs:
            self.init_guess_bra = kwargs['init_guess_bra']
        if "init_guess_ket" in kwargs:
            self.init_guess_ket = kwargs['init_guess_ket']
        if "init_guess" in kwargs:
            self.init_state = kwargs["init_guess"]
        self.BK.kernel()
        self.opt_res = deepcopy(self.BK.opt_res)
        self.opt_res.x = [-2*i for i in self.opt_res.x] #translating to tq
        return self.BK.opt_res.e

    def simulate(self,params:Union[list,dict])->float:
        if isinstance(params,Variables):
            params = params.store
        if isinstance(params,dict):
            v: dict = deepcopy(self.variables)
            v.update(params)
            tvars: list = deepcopy(self.BK.total_variables)
            params:list = [map_variables(x,v) for x in tvars]
        params:list = [-i/2 for i in params] #Here translation 
        return self.BK.expval(angles=params)

    @property
    def energy(self)->float:
        return self.opt_res.e

    @property
    def bra(self):
        """
        Excitation operators applied to the bra.
        """
        return self.BK.ex_ops_bra 
    
    @bra.setter
    def bra(self, bra):
        '''
        Expected indices in [[(0,2),(1,3),...],[(a,b),(c,d),...],...] Format (Upthendown order)
        '''
        bra,params_bra,_ = translate_indices(bra)
        if self.variables_bra is None:
            self.variables_bra = params_bra
        self.BK.ex_ops_bra = bra
    
    @property
    def ket(self):
        """
        Excitation operators applied to the ket.
        """
        return self.BK.ex_ops_ket 
    
    @ket.setter
    def ket(self, ket) -> None:
        '''
        Expected indices in [[(0,2),(1,3),...],[(a,b),(c,d),...],...] Format (Upthendown order)
        '''
        ket,params_ket,_ = translate_indices(ket)
        if self.variables_ket is None: 
            self.variables_ket = params_ket
        self.BK.ex_ops_ket = ket

    @property
    def variables_bra(self) -> dict:
        """Tequila Circuit Bra variables."""
        bar:dict = deepcopy(self.BK.var_to_param_bra)
        if bar is not None:
            for i in bar.keys(): #Here to tequila
                if isinstance(bar[i],Number):
                    bar[i] = -2*(bar[i])
        return bar 
    
    @property
    def params_bra(self):
        """TCC Circuit Bra parameters (values after minimization or variables name)."""
        return self.BK.params_bra
    
    @variables_bra.setter
    def variables_bra(self, variables_bra):
        '''
        See TCC variables
        '''
        self.BK.params_bra = variables_bra

    @property
    def variables_ket(self):
        """Tequila circuit Ket parameters."""
        bar = deepcopy(self.BK.var_to_param_ket)
        if bar is not None:
            for i in bar.keys(): #to tequila
                if isinstance(bar[i],Number):
                    bar[i] = -2*(bar[i])
        return bar
    
    @property
    def params_ket(self):
        """TCC Circuit Ket parameters (values after minimization or variables name)."""
        return self.BK.params_ket
    
    @variables_ket.setter
    def variables_ket(self, variables_ket):
        '''
        See TCC variables
        '''
        self.BK.params_ket = variables_ket
    
    @property
    def variables(self) -> dict:
        """Tequila circuit variables."""
        bar:dict = deepcopy(self.BK.var_to_param)
        if bar is not None:
            for i in bar.keys(): #Here to tequila 
                if isinstance(bar[i],Number):
                    bar[i] = -2*(bar[i])
        return bar 
    
    @property
    def params(self):
        """TCC parameters (values after minimization or variables name)."""
        return [i for i in self.BK.params if i is not None]
    
    @variables.setter
    def variables(self, variables):
        """Tequila circuit variables."""
        self.BK.params = variables

    @property
    def init_state_bra(self):
        """
        The circuit initial state before applying the excitation operators. Usually RHF.

        See Also
        --------
        BK.get_init_state_dataframe: Returns initial state information dataframe.
        """
        return self.BK.init_state_bra
    
    @property
    def init_state_ket(self):
        """
        The circuit initial state before applying the excitation operators. Usually RHF.

        See Also
        --------
        BK.get_init_state_dataframe: Returns initial state information dataframe.
        """
        return self.BK.init_state_ket
    
    @property
    def init_state(self):
        """
        The circuit initial state before applying the excitation operators. Usually RHF.

        See Also
        --------
        BK.get_init_state_dataframe: Returns initial state information dataframe.
        """
        return self.BK.init_state_bra ,self.BK.init_state_ket

    @init_state_bra.setter
    def init_state_bra(self, init_state_bra):
        self.BK.init_state_bra = init_state_bra
    
    @init_state_ket.setter
    def init_state_ket(self, init_state_ket):
        self.BK.init_state_ket  = init_state_ket

    @init_state.setter
    def init_state(self, init_state):
        self.BK.init_state_bra  = init_state
        self.BK.init_state_ket  = init_state

    @property
    def init_guess(self):
        """
        Initial Angle Value for minimization, default all 0.
        """
        return [-2*i for i in self.BK.init_guess]
    
    @init_guess.setter
    def init_guess(self, init_guess):
        self.BK.init_guess = [-i/2 for i in init_guess]

def init_state_from_wavefunction(wvf:QubitWaveFunction):
    if not isinstance(wvf._state,dict):
        return init_state_from_array(wvf=wvf)
    init_state = []
    for i in wvf._state:
        vec = bin(i)[2:]
        if len(vec) < wvf.n_qubits:
            vec = '0'*(wvf.n_qubits-len(vec))+vec
        init_state.append([vec,wvf._state[i].real])#tcc automatically does this, but with an anoying message everytime
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
            init_state.append([vec,idx.real]) #tcc automatically does this, but with an anoying message everytime
    return init_state

def translate_indices(indices):
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