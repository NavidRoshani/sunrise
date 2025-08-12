import typing
import numbers
from tequila.objective.objective import FixedVariable,Variable
from .circuit import FCircuit
from .fgateimpl import *

def FermionicExcitation(indices:typing.Union[list,tuple]|None=None, variables:typing.Union[typing.Hashable, numbers.Real, Variable, FixedVariable]|None=None, reordered:bool=False)->FCircuit:
    return FCircuit.wrap_gate(FermionicExcitationImpl(indices,variables,reordered))

def UR(i:int,j:int, variables:typing.Union[typing.Hashable, numbers.Real, Variable, FixedVariable]|None=None)->FCircuit:
    return FCircuit.wrap_gate(URImpl(i,j,variables))

def UC(i:int,j:int, variables:typing.Union[typing.Hashable, numbers.Real, Variable, FixedVariable]|None=None)->FCircuit:
    return FCircuit.wrap_gate(UCImpl(i,j,variables))

def UX(indices:typing.Union[list,tuple]|None=None, variables:typing.Union[typing.Hashable, numbers.Real, Variable, FixedVariable]|None=None, reordered:bool=False)->FCircuit:
    return FCircuit.wrap_gate(FermionicExcitationImpl(indices,variables,reordered))

