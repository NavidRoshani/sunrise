from sunrise.expval.tequila_expval import TequilaBraket

SUPPORTED_FERMIONIC_BACKENDS = ["tequila", "fqe", "tcc"]
INSTALLED_FERMIONIC_BACKENDS = {}#{"tequila": TequilaBraket}

try:
    from sunrise.expval.tcc_expval import TCCBraket
    INSTALLED_FERMIONIC_BACKENDS["tcc"] = TCCBraket
except ImportError:
    pass
try:
    from sunrise.expval.fqe_expval import FQEBraKet
    INSTALLED_FERMIONIC_BACKENDS["fqe"] = FQEBraKet
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
    bra/ket/circuit: FCircuit
        Fermionic Circuit containing the excitations, variables and initial state.
    (Optional) backend_kwargs:dict
        engine
            see tcc -> UCC engine
        dtype
            see tcc.set_dtype
        backend
            see tcc.set_backend
        else: kwargs provided to the UCC object initialization
    '''
    #any kwargs and circuit form should be managed inside each class
    return INSTALLED_FERMIONIC_BACKENDS[backend.lower()](*args,**kwargs) 
