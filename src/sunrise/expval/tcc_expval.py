import tencirchem as tcc
from tequila import QCircuit,QubitHamiltonian,TequilaException,Molecule,QubitWaveFunction,simulate
from tequila.quantumchemistry import QuantumChemistryPySCF
from tequila.quantumchemistry import qc_base
from numbers import Number
from tensorcircuit import Circuit
from numpy import array
#assumed to be installed pyscf since dependency for sunrise and tcc

def TCCBraket(molecule:qc_base=None,U:list[QCircuit,list,tuple]=None,bra:list[QCircuit,array]=None,ket:list[QCircuit,array]=None,reference:list[QCircuit,QubitWaveFunction,str,Circuit]=None,operator:QubitHamiltonian=None,tcc_args:dict=None,*args,**kwargs)->tcc.UCC:
    if 'backend' in  tcc_args:
        tcc.set_backend(tcc_args['backend'])
        tcc_args.pop('backend')
    if 'dtype' in tcc_args:
        tcc.set_dtype(tcc_args['dtype'])
        tcc_args.pop('dtype')
    #add here more options
    kwargs.update(tcc_args)
    if 'circuit' in kwargs:
        circuit = kwargs['circuit']
        kwargs.pop('circuit')
        if U is not None:
            raise TequilaException('Two circuits provided?')
        else:
            U = circuit
    if U is not None and isinstance(U, (list, tuple)):
        if isinstance(U[0], Number):
            U = [[U, ], ]
        elif isinstance(U[0][0], Number):
            U = [U, ]
    if U is not None and ket is not None:
        raise TequilaException('Two circuit provided?')
    elif U is not None:
        ket = U

    if molecule is not None:
        molecule = QuantumChemistryPySCF.from_tequila(molecule)
        UCC = tcc.UCC(mol=molecule.pyscf_molecule,run_hf= False, run_mp2= False, run_ccsd= False, run_fci= False,**kwargs)
    elif 'integral_manager' in kwargs and 'parameters' in kwargs:
        integral = kwargs['integral_manager']
        params = kwargs['parameters']
        molecule = Molecule(parameters=params,integral_manager=integral,backend='pyscf')
        UCC = tcc.UCC(mol=molecule.pyscf_molecule,run_hf= False, run_mp2= False, run_ccsd= False, run_fci= False,**kwargs)
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
        if 'n_elec' in kwargs:
            n_elec = kwargs['n_elec']
            kwargs.pop('n_elec')
        elif 'n_electrons' in kwargs:
            n_elec = kwargs['n_electrons']
            kwargs.pop('n_electrons')
        if 'e_core' in kwargs:
            e_core = kwargs['e_core']
            kwargs.pop('e_core')
        elif 'constant_term' in kwargs:
            e_core = kwargs['constant_term']
            kwargs.pop('constant_term')
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
        if all([i is not None for i in[int2e,int1e,n_elec,e_core,mo_coeff]]):
            UCC = tcc.UCC.from_integral(int1e=int1e, int2e=int2e,n_elec= n_elec, e_core=e_core,ovlp=ovlp,run_hf= False, run_mp2= False, run_ccsd= False, run_fci= False,**kwargs)
        else:
            raise TequilaException('Not enough molecular data provided')

    if reference is not None:
        if isinstance(reference,QCircuit):
            if 'variables' in kwargs:
                variables = kwargs['variables']
                kwargs.pop('variables')
            else: variables = {}
            init_state = from_wavefunction(simulate(reference,variables=variables))
            for i in init_state:
                i[0] = UCC.get_addr(i[0])
            init = [0] * UCC.civector_size
            for i in init_state:
                init[i[0]] = i[1]
            UCC.init_state = init
        elif isinstance(reference,str):
            init_state = from_wavefunction(QubitWaveFunction.from_string(reference))
            for i in init_state:
                i[0] = UCC.get_addr(i[0])
            init = [0] * UCC.civector_size
            for i in init_state:
                init[i[0]] = i[1]
            UCC.init_state = init
        elif isinstance(reference,QubitWaveFunction):
            init_state = from_wavefunction(reference)
            for i in init_state:
                i[0] = UCC.get_addr(i[0])
            init = [0] * UCC.civector_size
            for i in init_state:
                init[i[0]] = i[1]
            UCC.init_state = init
        elif isinstance(reference,Circuit):
            UCC.init_state = reference.state()
        else:
            UCC.init_state = reference

    if 'H' in kwargs:
        H = kwargs['H']
        kwargs.pop('H')
        operator = H
        #TODO COMO HACER BRAKETS TODAS OPCIONES
    if operator is None:
        # raise TequilaException(
        #     "Not Operator other than the Molecular Hamiltonian defined when creating the UCC object is supported by TCC")
        return UCC.civector().conj() @ UCC.civector()
    return UCC

def from_wavefunction(wvf):
    init_state = []
    for i in wvf._state:
        vec = bin(i)[2:]
        vup = ''
        vdw = ''
        for j in range(len(vec)//2):
            vup += vec[2*j]
            vdw += vec[2*j+1]
        vec = (vup + vdw)[::-1]
        init_state.append([vec,wvf._state[i]])
    return init_state
