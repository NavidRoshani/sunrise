import tencirchem as tcc
from sunrise.expval.tcc_engine.braket import EXPVAL
from tequila import QCircuit,TequilaException,Molecule,QubitWaveFunction,simulate,Variable,Objective
from tequila.objective.objective import Variables
from tequila.quantumchemistry import qc_base
from tequila.utils.bitstrings import BitString
from tensorcircuit import Circuit
from numbers import Number
from numpy import array,ceil,pi
from pyscf.gto import Mole
from sunrise.expval.pyscf_molecule import from_tequila
from copy import deepcopy
#assumed to be installed pyscf since dependency for sunrise and tcc

class TCCBraket:
    def __init__(self,bra:list=None,ket:list=None,init_state_bra:list[QCircuit,QubitWaveFunction,Circuit,str]=None,
                 init_state_ket:list[QCircuit,QubitWaveFunction,Circuit,str]=None,tcc_kwargs:dict={},
                 variables_bra:any=None,variables_ket:any=None,*args,**kwargs):
        if 'n_qubits' in kwargs:
            self.n_qubits = kwargs['n_qubits']
            kwargs.pop('n_qubits')
        elif 'molecule' in kwargs and kwargs['molecule'] is not None:
            if isinstance(kwargs['molecule'],qc_base.QuantumChemistryBase):
                self.n_qubits = 2*kwargs['molecule'].n_orbitals
            elif isinstance(kwargs['molecule'],Mole):
                self.n_qubits = 2*kwargs['molecule'].mo_coeff.shape[0]
        elif 'integral_manager' in kwargs:
            self.n_qubits = len(kwargs['integral_manager'].active_orbitals)
        elif 'int1e' in kwargs:
            self.n_qubits = 2*len(kwargs['int1e'])
        else: self.n_qubits = None

        if 'engine' in tcc_kwargs:
            engine = tcc_kwargs['engine']
            tcc_kwargs.pop('engine')
        else: engine = None

        if 'backend' in  tcc_kwargs:
            tcc.set_backend(tcc_kwargs['backend'])
            tcc_kwargs.pop('backend')
        if 'dtype' in tcc_kwargs:
            tcc.set_dtype(tcc_kwargs['dtype'])
            tcc_kwargs.pop('dtype')
        
        if 'circuit' in kwargs:
            circuit = kwargs['circuit']
            kwargs.pop('circuit')
            if ket is not None:
                raise TequilaException('Two circuits provided?')
            else:
                ket = circuit
        if 'indices' in kwargs:
            indices = kwargs['indices']
            kwargs.pop('indices')
            if indices is not None: 
                if ket is not None:
                    raise TequilaException('Two circuits provided?')
                else:
                    ket = indices
        if ket is not None and isinstance(ket, (list, tuple)):
            if isinstance(ket[0], Number):
                ket = [[ket, ], ]
            elif isinstance(ket[0][0], Number):
                ket = [ket, ]
        if bra is not None:
            if isinstance(bra, (list, tuple)):
                if isinstance(bra[0], Number):
                    bra = [[bra, ], ]
                elif isinstance(bra[0][0], Number):
                    bra = [bra, ]
        
        if 'init_state' in kwargs:
            init_state  = kwargs['init_state']
            kwargs.pop('init_state')
            if isinstance(init_state,QCircuit):
                if self.n_qubits is not None:
                    init_state.n_qubits = self.n_qubits
                if 'init_state_variables' in kwargs:
                    ivariables = kwargs['init_state_variables']
                    kwargs.pop('init_state_variables')
                else: ivariables = {}
                res = simulate(init_state,variables=ivariables)
                init_state = init_state_from_wavefunction(res)
                n_elec = init_state[0][0].count('1')
            elif isinstance(init_state,str):
                init_state = init_state_from_wavefunction(QubitWaveFunction.from_string(init_state))
                n_elec = init_state[0][0].count('1')
            elif isinstance(init_state,QubitWaveFunction):
                init_state = init_state_from_wavefunction(init_state)
                n_elec = init_state[0][0].count('1')
            elif isinstance(init_state,Circuit):
                init_state._nqubits = self.n_qubits
                init_state = init_state.state()
                n_elec = [bin(i) for i in range(len(init_state)) if init_state[i].real>1.e-6][0].count('1')
            elif isinstance(init_state_ket,(list,array)):
                n_elec = [bin(i) for i in range(len(init_state)) if init_state[i].real>1.e-6][0].count('1')
            else:
                try:
                    init_state = init_state_from_wavefunction(QubitWaveFunction.convert_from(val=init_state,n_qubits=self.n_qubits))
                    n_elec = init_state[0][0].count('1')
                except:
                    raise TequilaException(f'Init_state format not recognized, provided {type(init_state).__name__}')
        else:
            n_elec = None
            init_state =None
        if init_state_bra is not None:
            if isinstance(init_state_bra,QCircuit):
                if self.n_qubits is not None:
                    init_state_bra.n_qubits = self.n_qubits
                if 'init_state_bra_variables' in kwargs:
                    ivariables = kwargs['init_state_bra_variables']
                    kwargs.pop('init_state_bra_variables')
                else: ivariables = {}
                init_state_bra = init_state_from_wavefunction(simulate(init_state_bra,variables=ivariables))
                n_e_bra = init_state_bra[0][0].count('1')
            elif isinstance(init_state_bra,str):
                init_state_bra = init_state_from_wavefunction(QubitWaveFunction.from_string(init_state_bra))
                n_e_bra = init_state_bra[0][0].count('1')
            elif isinstance(init_state_bra,QubitWaveFunction):
                init_state_bra = init_state_from_wavefunction(init_state_bra)
                n_e_bra = init_state_bra[0][0].count('1')
            elif isinstance(init_state_bra,Circuit):
                init_state_bra._nqubits = self.n_qubits
                init_state_bra = init_state_bra.state()
                n_e_bra = [bin(i) for i in range(len(init_state_bra)) if init_state_bra[i].real>1.e-6][0].count('1')
            elif isinstance(init_state_bra,(list,array)):
                n_e_bra = [bin(i) for i in range(len(init_state_bra)) if init_state_bra[i].real>1.e-6][0].count('1')
            else:
                try:
                    init_state_bra = init_state_from_wavefunction(QubitWaveFunction.convert_from(val=init_state_bra,n_qubits=self.n_qubits))
                    n_e_bra = init_state_bra[0][0].count('1')
                except:
                    raise TequilaException(f'Init_state_bra format not recognized, provided {type(init_state_bra).__name__}')
        else:
            n_e_bra = None
        if init_state_ket is not None:
            if isinstance(init_state_ket,QCircuit):
                if self.n_qubits is not None:
                    init_state_ket.n_qubits = self.n_qubits
                if 'init_state_ket_variables' in kwargs:
                    ivariables = kwargs['init_state_ket_variables']
                    kwargs.pop('init_state_ket_variables')
                else: ivariables = {}
                init_state_ket = init_state_from_wavefunction(simulate(init_state_ket,variables=ivariables))
                n_e_ket = init_state_ket[0][0].count('1')
            elif isinstance(init_state_ket,str):
                init_state_ket = init_state_from_wavefunction(QubitWaveFunction.from_string(init_state_ket))
                n_e_ket = init_state_ket[0][0].count('1')
            elif isinstance(init_state_ket,QubitWaveFunction):
                init_state_ket = init_state_from_wavefunction(init_state_ket)
                n_e_ket = init_state_ket[0][0].count('1')
            elif isinstance(init_state_ket,Circuit):
                init_state_ket._nqubits = self.n_qubits
                init_state_ket = init_state_ket.state()
                n_e_ket = [bin(i) for i in range(len(init_state_ket)) if init_state_ket[i].real>1.e-6][0].count('1')
            elif isinstance(init_state_ket,(list,array)):
                n_e_ket = [bin(i) for i in range(len(init_state_ket)) if init_state_ket[i].real>1.e-6][0].count('1')
            else:
                try:
                    init_state_ket = init_state_from_wavefunction(QubitWaveFunction.convert_from(val=init_state_ket,n_qubits=self.n_qubits))
                    n_e_ket = init_state_ket[0][0].count('1')
                except:
                    raise TequilaException(f'Init_state_ket format not recognized, provided {type(init_state_ket).__name__}')
        else: n_e_ket = None
        if all(i is None for i in [n_elec,n_e_bra,n_e_ket]):
            if 'n_elec' in kwargs:
                n_elec = kwargs['n_elec']
                kwargs.pop['n_elec']
            elif 'n_electrons' in kwargs:
                n_elec = kwargs['n_electrons']
                kwargs.pop('n_electrons')
            elif 'molecule' in kwargs:
                if isinstance(kwargs['molecule'],qc_base.QuantumChemistryBase):     
                    n_elec = kwargs['molecule'].n_electrons
                else:
                    n_elec = kwargs['molecule'].nelectron
            elif 'parameters' in kwargs:
                n_elec = kwargs['parameters'].n_electrons
            else:
                raise TequilaException("No way of defining the amount of electrons provided")
        if n_elec is None:   
            if n_e_bra is not None:
                assert n_e_bra == n_e_ket
            n_elec = n_e_ket
            
        if 'molecule' in kwargs and kwargs['molecule']:
            molecule = kwargs['molecule']
            kwargs.pop('molecule')
            if isinstance(molecule,qc_base.QuantumChemistryBase):
                molecule = from_tequila(molecule)
            elif isinstance(molecule,Mole):
                self.n_qubits = 2*molecule.mo_coeff.shape[0]
            self.BK = EXPVAL(mol=molecule,run_hf= False, run_mp2= False, run_ccsd= False, run_fci= False,init_method="zeros",**tcc_kwargs)
        elif 'integral_manager' in kwargs and 'parameters' in kwargs:
            integral = kwargs['integral_manager']
            params = kwargs['parameters']
            kwargs.pop('integral_manager')
            kwargs.pop('parameters')
            molecule = Molecule(parameters=params,integral_manager=integral)
            self.n_qubits = 2*molecule.n_orbitals
            molecule = from_tequila(molecule)
            self.BK = EXPVAL(mol=molecule,run_hf= False, run_mp2= False, run_ccsd= False, run_fci= False,init_method="zeros",**tcc_kwargs)
        else:
            int1e = None
            int2e = None
            n_elec = None
            e_core = None
            mo_coeff = None
            ovlp = None
            if "int1e"  in kwargs:
                int1e = kwargs['int1e']
                kwargs.pop('int1e')
            elif "one_body_integrals"  in kwargs:
                int1e = kwargs['one_body_integrals']
                kwargs.pop('one_body_integrals')
            if 'int2e' in kwargs:
                int2e = kwargs['int2e']
                kwargs.pop('int2e')
            elif 'two_body_integrals' in kwargs:
                int2e = kwargs['two_body_integrals']
                kwargs.pop('two_body_integrals')
            if 'e_core' in kwargs:
                e_core = kwargs['e_core']
                kwargs.pop('e_core')
            elif 'constant_term' in kwargs:
                e_core = kwargs['constant_term']
                kwargs.pop('constant_term')
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
            if 'n_elec' in kwargs:
                n_elec=kwargs['n_elec']
                kwargs.pop('n_elec')
            if all([i is not None for i in[int2e,int1e,mo_coeff]]):
                self.BK = EXPVAL.from_integral(int1e=int1e, int2e=int2e,n_elec= n_elec, e_core=e_core,ovlp=ovlp,mo_coeff=mo_coeff,init_method="zeros",run_hf= False, run_mp2= False, run_ccsd= False, run_fci= False,**tcc_kwargs)
            else:
                raise TequilaException('Not enough molecular data provided')

        if init_state_bra is not None: #init state initialization splitted bcs we need BK structe, but we get info to initialize BK from init state
            if isinstance(init_state_bra[0],Number):
                self.BK.init_state_bra = init_state_bra
            else:
                for i in init_state_bra:
                    i[0] = self.BK.get_addr(i[0])
                init = [0] * self.BK.civector_size
                for i in init_state_bra:
                    init[i[0]] = i[1]
                self.init_state_bra = init
        if init_state_ket is not None: #TCC builds HF if None provided
            if isinstance(init_state_ket[0],Number):
                self.init_state_ket = init_state_ket 
            else:
                for i in init_state_ket:
                    i[0] = self.BK.get_addr(i[0])
                init = [0] * self.BK.civector_size
                for i in init_state_ket:
                    init[i[0]] = i[1]
                self.init_state_ket = init
        if init_state is not None: #TCC builds HF if None provided
            if isinstance(init_state[0],Number):
                self.init_state_ket = init_state_ket 
            else:
                for i in init_state:
                    i[0] = self.BK.get_addr(i[0])
                init = [0] * self.BK.civector_size
                for i in init_state:
                    init[i[0]] = i[1]
                self.init_state_ket = init
        
        if 'variables' in kwargs:
            v = kwargs['variables']
            kwargs.pop('variables')
            if variables_ket is None:
                self.variables_ket =v
            elif variables_bra is not None:
                raise TequilaException("Too many variables, provided Variables, variables_ket and variables_bra?")
            else: self.variables_bra = v

        if variables_ket is not None:
            self.variables_ket = variables_ket
        if variables_bra is not None:
            self.variables_bra = variables_bra

        if ket is not None:
            self.ket = ket
        if bra is not None:
            self.bra = bra
        
        if 'init_guess_ket' in kwargs:
            self.init_guess_ket = kwargs['init_guess_ket']
            kwargs.pop('init_guess_ket')
        if 'init_guess_bra' in kwargs:
            self.init_guess_bra = kwargs['init_guess_bra']
            kwargs.pop('init_guess_bra')
        
        if 'engine' in kwargs:
            engine = kwargs['engine']
            kwargs.pop('engine')
        if engine is not None:
            self.BK.engine = engine
        self.opt_res = None
        
    def minimize(self,**kwargs):
        if 'init_guess_bra' in kwargs:
            self.init_guess_bra = kwargs['init_guess_bra']
        if "init_guess_ket" in kwargs:
            self.init_guess_ket = kwargs['init_guess_ket']
        if "init_guess" in kwargs:
            self.init_state = kwargs["init_guess"]
        self.BK.kernel()
        self.opt_res = deepcopy(self.BK.opt_res)
        self.opt_res.x = [2*i for i in self.opt_res.x] #translating to tq
        return self.BK.opt_res.e

    def simulate(self,params:list[list,dict]):
        if isinstance(params,Variables):
            params = params.store
        if isinstance(params,dict):
            v = deepcopy(self.variables)
            v.update(params)
            tvars = deepcopy(self.BK.total_variables)
            params = [map_variables(x,v) for x in tvars]
        params = [i/2 for i in params] #Here translation 
        return self.BK.expval(angles=params)

    @property
    def energy(self):
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
        Expected indices in [[(0,2),(1,3),...],[(a,b),(c,d),...],...] Format (udud order, being 0 the lowest energy MO)
        '''
        bra,params_bra,_ = from_indices(bra,len(self.BK.aslst))
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
    def ket(self, ket):
        '''
        Expected indices in [[(0,2),(1,3),...],[(a,b),(c,d),...],...] Format (udud order, being 0 the lowest energy MO)
        '''
        ket,params_ket,_ = from_indices(ket,len(self.BK.aslst))
        if self.variables_ket is None: 
            self.variables_ket = params_ket
        self.BK.ex_ops_ket = ket

    @property
    def variables_bra(self):
        """Tequila Circuit Bra variables."""
        bar = deepcopy(self.BK.var_to_param_bra)
        if bar is not None:
            for i in bar.keys(): #Here to tequila
                if isinstance(bar[i],Number):
                    bar[i] = 2*bar[i]
        return bar 
    
    @property
    def params_bra(self):
        """TCC Circuit Bra parameters (values after minimization or variables name). If value, they are in tcc periodicity (tq/2=tcc)"""
        return [self.BK.params_bra]
    
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
                    bar[i] = 2*bar[i]
        return bar
    
    @property
    def params_ket(self):
        """TCC Circuit Ket parameters (values after minimization or variables name). If value, they are in tcc periodicity (tq/2=tcc)"""
        return self.BK.params_ket
    
    @variables_ket.setter
    def variables_ket(self, variables_ket):
        '''
        See TCC variables
        '''
        self.BK.params_ket = variables_ket
    
    @property
    def variables(self):
        """Tequila circuit variables."""
        bar = deepcopy(self.BK.var_to_param)
        if bar is not None:
            for i in bar.keys(): #Here to tequila 
                if isinstance(bar[i],Number):
                    bar[i] = 2*bar[i]
        return bar 
    
    @property
    def params(self):
        """TCC parameters (values after minimization or variables name). If value, they are in tcc periodicity (tq/2=tcc)"""
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
        return [i for i in self.BK.init_guess]
    
    @init_guess.setter
    def init_guess(self, init_guess):
        self.BK.init_guess = [i for i in init_guess]

def init_state_from_wavefunction(wvf:QubitWaveFunction):
    if not isinstance(wvf._state,dict):
        return init_state_from_array(wvf=wvf)
    init_state = []
    for i in wvf._state:
        vec = bin(i)[2:]
        vup = ''
        vdw = ''
        for j in range(len(vec)//2):
            vup += vec[2*j]
            vdw += vec[2*j+1]
        vec = (vup + vdw)[::-1]
        init_state.append([vec,wvf._state[i].real])#tcc automatically does this, but with an anoying message everytime
    return init_state

def init_state_from_array(wvf:QubitWaveFunction,tol=1e-6):
    if isinstance(wvf._state,dict):
        return init_state_from_wavefunction(wvf)
    init_state = []
    nq = wvf.n_qubits
    nq = int(2* (ceil(nq/2))) #if HCB may be odd amount of qubits
    for i,idx in enumerate(wvf._state):
        vec = BitString.from_int(i)
        vec.nbits = nq
        vec = vec.binary
        vup = ''
        vdw = ''
        for j in range(len(vec)//2):
            vup += vec[2*j]
            vdw += vec[2*j+1]
        vec = (vup + vdw)
        if abs(idx) > tol:
            init_state.append([vec,idx.real]) #tcc automatically does this, but with an anoying message everytime
    return init_state

def from_indices(indices,nmo):
    '''
    Expected indices like [[(0,2),(1,3),...],[(a,b),(c,d),...],...]
    Returned [(0,0+nno,...,1+nmo,2),(a,c,...,d,b)]
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
            exc.append(idx[0]//2+(idx[0]%2)*nmo)
            exc.insert(0,idx[1]//2+(idx[1]%2)*nmo)
        ex_ops.append(tuple(exc))
    return ex_ops,params,param_ids

def map_variables(x:list[Variable,Objective],dvariables:dict):
    if isinstance(x,Variable):
        x = x.map_variables(dvariables)
    elif isinstance(x,Objective):
        x=simulate(x,dvariables)
    return x