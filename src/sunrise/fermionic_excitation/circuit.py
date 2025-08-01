import tequila as tq
from tequila import QCircuit,QubitWaveFunction,Variable,TequilaException,TequilaWarning
from tequila.circuit.gates import QubitExcitationImpl
from tequila.circuit._gates_impl import assign_variable
from tequila.quantumchemistry.chemistry_tools import FermionicGateImpl
from tequila.objective.objective import FixedVariable
import typing
import numbers
import warnings
from copy import deepcopy
from numpy import ndarray

class FCircuit:
    def __init__(self, operations:typing.Union[list,tuple]|None=None,initial_state:typing.Union[QCircuit,QubitWaveFunction,str,int]|None=None,init_state_variables:dict|None=None):
        self._operations = operations if operations is not None else []
        if initial_state is None or isinstance(initial_state,QubitWaveFunction):
            pass
        elif isinstance(initial_state,QCircuit):
            initial_state = tq.simulate(initial_state,variables=init_state_variables)
        elif isinstance(initial_state,str):
            initial_state = QubitWaveFunction.from_string(initial_state)
        elif isinstance(initial_state,(list,ndarray)):
            initial_state = QubitWaveFunction.from_array(initial_state)
        else:
            try:
                initial_state = QubitWaveFunction.convert_from(val=initial_state,n_qubits=self.n_qubits)
            except:
                raise TequilaException(f'Init_state format not recognized, provided {type(initial_state).__name__}')
        self._initial_state:QubitWaveFunction = initial_state 
        self.verify()

    def __add__(self, other):
        initial_state = self._initial_state
        operations = self._operations
        if hasattr(other, "_operations"):
            if self._initial_state is None:
                initial_state = other._initial_state
            elif other._initial_state != self._initial_state:
                raise TequilaException(f"FermionicCircuit + FermionicCircuit with two different initial states:\n{self._initial_state}, {other._initial_state}")
            return FCircuit(operations=operations, initial_state=initial_state)
        elif isinstance(other,QCircuit):
            other = self.from_Qcircuit(other)
            return self.__add__(other)
        raise TequilaException(f'Fermionic Circuit expected, received {type(other).__name__}')
    
    def __iadd__(self,other):
        initial_state = self._initial_state
        if hasattr(other, "_operations"):
            if self._initial_state is None:
                initial_state = other._initial_state
            elif other._initial_state != self._initial_state:
                raise TequilaException(f"Fermionic Circuit + Fermionic Circuit with two different initial states:\n{self._initial_state}, {other._initial_state}")
            self._operations.extend(other._operations)
            self._initial_state = initial_state
            return self
        elif isinstance(other,QCircuit):
            other = self.from_Qcircuit(other)
            return self.__iadd__(other)
        else:
            raise TequilaException(f'Fermionic Circuit expected, received {type(other).__name__}')       

    def __str__(self):
        result = "Fermionic Circuit: \n"
        if self._initial_state is not None:
            result += str(QubitWaveFunction.from_array(self._initial_state)) +'\n'
        if self._operations is not None:
            for op in self._operations:
                result += f'Excitation{op[1]} variable = {op[0]}' + "\n"
        return result

    def verify(self):
        operations = [[tq.assign_variable(x[0]), tuple(x[1])] for x in self._operations]
        self._operations = operations

    def extract_variables(self)->list[Variable]:
        return [op[0] for op in self._operations]

    @property
    def variables(self)->list[Variable]:
        return self.extract_variables()
    
    @property
    def excitations(self)->list:
        l = []
        for exc in self._operations:
            l.append(exc[1])
        return l

    @property
    def n_qubits(self)->int:
        if self._initial_state is not None:
            return self._initial_state.n_qubits
        elif self._operations is not None:
            big = 0
            for exct in self.excitations:
                for ex in exct:
                    big = [max(big,idx) for idx in ex][-1]
            return big
        else: return 0

    @classmethod
    def from_Qcircuit(cls,circuit:QCircuit,**kwargs):
        operations = []
        reference = QCircuit()
        begining = True
        if 'variables' in kwargs:
            circuit = circuit.map_variables(kwargs['variables'])
            kwargs.pop("variables")
        for gate in circuit.gates:
            if begining and not hasattr(gate,'_parameter') or isinstance(gate._parameter,numbers.Number):
                reference += gate
            elif isinstance(gate,QubitExcitationImpl): #maybe we can consider other gates but basic implementation for the moment
                if isinstance(gate._parameter,numbers.Number):
                    reference += gate
                elif isinstance(gate,FermionicGateImpl):
                    begining = False
                    operations.append[gate.parameter,gate.indices]
                else:
                    temp = []
                    for i in range(len(gate._target)//2):
                        temp.append((gate._target[2*i],gate._target[2*i+1]))
                    operations.append([gate.parameter,temp])
            else:
                raise TequilaException(f'Gate {gate._name}({gate._parameter}) not allowed')
        return cls(operations=operations,initial_state=reference)

    def map_variables(self, variables: dict, *args, **kwargs):
        """

        Parameters
        ----------
        variables
            dictionary with old variable names as keys and new variable names or values as values
        Returns
        -------
        Circuit with changed variables

        """

        variables = {assign_variable(k): assign_variable(v) for k, v in variables.items()}

        # failsafe
        my_variables = self.extract_variables()
        for k, v in variables.items():
            if k not in my_variables:
                warnings.warn(
                    "map_variables: variable {} is not part of circuit with variables {}".format(k, my_variables),
                    TequilaWarning,
                )

        new_operations = [[deepcopy(op[0]).map_variables(variables),op[1]] for op in self._operations]

        return FCircuit(operations=new_operations,initial_state=self._initial_state)


def make_excitation_gate(indices:list, angle:typing.Union[typing.Hashable, numbers.Real, Variable, FixedVariable], *args, **kwargs)->FCircuit:
    return FCircuit([(angle,indices),])

def UR(i:int,j:int,angle:typing.Union[typing.Hashable, numbers.Real, Variable, FixedVariable])->FCircuit:
    return FCircuit(operations=[(angle,[(2*i,2*j)]),(angle,[(2*i+1,2*j+1)])])

def UC(i:int,j:int,angle:typing.Union[typing.Hashable, numbers.Real, Variable, FixedVariable])->FCircuit:
    U = FCircuit(operations=[(angle,[(2*i,2*j),(2*i+1,2*j+1)]),])
    return U

def UX(indices:list, angle:typing.Union[typing.Hashable, numbers.Real, Variable, FixedVariable], *args, **kwargs)->FCircuit:
    return FCircuit((angle,indices))


if __name__ == '__main__':
    U = FCircuit()
    U += FCircuit(operations=[("a",[(0,2),(1,3)])])
    U += FCircuit(operations=[("b",[(0,4),(1,5)])])
    U += UC(0,1,"c")
    U += UR(0,1,"r")
    U += make_excitation_gate([(2,4),(3,5)],'e')
    print(U)
    print(U.excitations)
    print(U.n_qubits)
    U = U.map_variables({'e':"eeeeeeee"})
    print(U)