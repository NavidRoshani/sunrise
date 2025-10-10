import tequila as tq
from tequila import BraKet,QCircuit,QubitHamiltonian,ExpectationValue
from tequila.quantumchemistry.chemistry_tools import NBodyTensor
from tequila import TequilaException
from tequila import QCircuit,TequilaException,Molecule
from tequila.quantumchemistry import qc_base
from numpy import argwhere
from pyscf.gto import Mole
from sunrise.expval.pyscf_molecule import MoleculeFromPyscf
from ..fermionic_excitation.circuit import FCircuit
from typing import Union

def TequilaBraket(bra:FCircuit|None=None,ket:FCircuit|None=None,operator:Union[str,QubitHamiltonian]='H',*args,**kwargs):

    if 'circuit' in kwargs:
        circuit = kwargs['circuit']
        kwargs.pop('circuit')
        if ket is not None:
            raise TequilaException('Two circuits provided?')
        else:
            ket = circuit

    if 'U' in kwargs:
        U = kwargs['U']
        kwargs.pop('U')
        if ket is not None:
            raise TequilaException('Two circuits provided?')
        else:
            ket = U


    molecule = None
    if 'molecule' in kwargs and kwargs['molecule']:
        molecule = kwargs['molecule']
        kwargs.pop('molecule')
        if isinstance(molecule,qc_base.QuantumChemistryBase):
            molecule = molecule
        elif isinstance(molecule,Mole):
            molecule = MoleculeFromPyscf(molecule=molecule)
    elif 'integral_manager' in kwargs and 'parameters' in kwargs:
        integral = kwargs['integral_manager']
        params = kwargs['parameters']
        kwargs.pop('integral_manager')
        kwargs.pop('parameters')
        molecule = Molecule(parameters=params,integral_manager=integral)
    else:
        int1e = None
        int2e = None
        e_core = None
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
        elif 'constant' in kwargs:
            e_core = kwargs['constant']
            kwargs.pop('constant')
        elif 'c' in kwargs:
            e_core = kwargs['c']
            kwargs.pop('c')    
        else: e_core = 0.
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
        if all([i is not None for i in[int2e,int1e]]):
            if isinstance(int2e,NBodyTensor):
                int2e = int2e.reorder('of').elems
            if 'transformation' in kwargs:
                trans = kwargs['transformation']
            else: trans = 'reordered-jordan-wigner'
            if 'molecule_arguments' in kwargs:
                molecule_arguments = kwargs['molecule_arguments']
            else: molecule_arguments = {}
            molecule = Molecule(one_body_integrals=int1e,two_body_integrals=int2e,constant_term=e_core,n_electrons=n_elec,transformation=trans,**molecule_arguments)
        else:
            raise TequilaException('Not enough molecular data provided')
    molecule.integral_manager.upthendown=True
    if ket is not None:
        ket = ket.to_upthendown(molecule.n_orbitals)
    if bra is not None:
        bra = bra.to_upthendown(molecule.n_orbitals)
    U0ket =  QCircuit()
    U0bra =  QCircuit()
    if isinstance(operator,str):
        if operator == 'H':
            operator = molecule.make_hamiltonian()
        elif operator == "HCB":
            operator = molecule.make_hardcore_boson_hamiltonian()
        else:
            operator = tq.paulis.from_string(operator)
    if bra is None or bra == ket:
        return ExpectationValue(U=ket.to_qcircuit(molecule=molecule),H=operator,*args,**kwargs)
    else:
        return BraKet(bra=bra.to_qcircuit(molecule=molecule),ket=ket.to_qcircuit(molecule=molecule),operator=operator,*args,**kwargs)
