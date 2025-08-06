import numbers
import typing
from tequila.objective.objective import FixedVariable,Variable
from .circuit import FCircuit
from tequila.circuit._gates_impl import assign_variable
from tequila import TequilaException
from copy import deepcopy

class FGateImpl:
    def __init__(self,indices:typing.Union[list,tuple]|None=None,variables:typing.Union[typing.Hashable, numbers.Real, Variable, FixedVariable]=None,reorederd:bool=False):
        self.reoredered=reorederd
        self._indices = indices
        self._variables=variables
        self._name = 'GenericFermionic'
        self.verify()
        return FCircuit.wrap_gate(gate=self)

    def reorder(self,norb:int)->typing.Union[tuple,list]:
        if not self.reoredered:
            self._indices = [[(idx[0]//2+(idx[0]%2)*norb,idx[1]//2+(idx[1]%2)*norb) for idx in gate] for gate in self._indices]
        return self
    
    def __str__(self):
        return f'{self._name}(Indices = {self.indices} Variable = {self.variables})'
    
    def __eq__(self, other):
        if self._name != other._name:
            return False
        elif self._indices != other._indices:
            return False
        elif self.extract_variables() != other.extract_variables():
            return False
        return True
            
    @property
    def indices(self):
        return self._indices
    
    @indices.setter
    def indices(self,indices):
        self._indices = indices
    
    def extract_variables(self)->list[Variable]:
        return [v for v in self.variables]

    @property
    def variables(self):
        return self._variables
    
    @variables.setter
    def variables(self,variables):
        self._variables = variables

    def verify(self):
        if isinstance(self._variables,(typing.Hashable, numbers.Real, Variable, FixedVariable)):
            self._variables = [self._variables] 
        self._variables = [assign_variable(v) for v in self._variables]
        if isinstance(self._indices[0],numbers.Number): #[1,3]
            self._indices = [[tuple(self._indices),],] #->[[(1,3)]]
        elif isinstance(self._indices[0][0],numbers.Number): #[(0,2),(1,2)]
            self._indices = [self._indices,] #->[[(0,2),(1,3)],]
        elif not isinstance(self._indices[0][0][0],numbers.Number): #[[(0,2),(1,2)]]
            raise TequilaException(f'Indices formating not recognized, received {self._indices}')
        assert len(self._variables) == len(self._indices)

    @property
    def qubits(self):
        if self._indices is not None:
            q = []
            for gate in self._indices:
                for exct in gate:
                    q.extend(exct)
            return sorted(list(set(q)))
        else: return 0

    @property
    def n_qubits(self):
        return len(self.qubits)
    
    def map_qubits(self,qubit_map:dict={}):
        self._indices =[[(qubit_map[idx[0]],qubit_map[idx[1]]) for idx in gate] for gate in self._indices]

    def map_variables(self,var_map:dict={}):
        self._variables = [var_map[v] for v in self._variables]

    def dagger(self):
        indinces = []
        variables = []
        cir = deepcopy(self)
        for gate in reversed(cir._indices):
            indinces.extend([[tuple([*idx][::-1]) for idx in gate]])
        cir.indices = indinces
        for v in cir._variables:
            variables.append(-1*v)
        cir.variables = variables
        return cir

    @property
    def max_qubit(self):
        if self.qubits:
            return self.qubits[-1]
        else: return 0

class FermionicExcitationImpl(FGateImpl):
    def __init__(self, indices:typing.Union[list,tuple]|None=None, variables:typing.Union[typing.Hashable, numbers.Real, Variable, FixedVariable]|None=None, reordered:bool=False):
        super().__init__(indices, variables, reordered)
        self._name = 'FermionicExcitation'

class URImpl(FGateImpl):
    def __init__(self, i:int,j:int, variables:typing.Union[typing.Hashable, numbers.Real, Variable, FixedVariable]|None=None):
        super().__init__([[(2*i,2*j)],[(2*i+1,2*j+1)]], [variables,variables], False)
        self._name = 'UR'
    def __str__(self):
        return  f'{self._name}(Indices = {self.indices} Variable = {self.extract_variables()})'
    
class UCImpl(FGateImpl):
    def __init__(self,i,j, variables:typing.Union[typing.Hashable, numbers.Real, Variable, FixedVariable]|None=None):
        super().__init__([[(2*i,2*j),(2*i+1,2*j+1)]], variables, False)
        self._name = 'UC'

class EdgesImpl(FGateImpl):
    def __init__(self, edges):
        indices = [[[(2*edge[0],2*edge[1]),(2*edge[0]+1,2*edge[1]+1)] for edge in edges]]
        variables = [str(edge) for edge in edges]
        super().__init__(indices, variables, False)
        self._name = 'Edges'
  