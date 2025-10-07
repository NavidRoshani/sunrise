
import tequila as tq
from tequila import simulate

class BraKetQulacs:

    def __init__(self, bra, ket, H):
        # translate tq -> qulacs
        if H is None:
            H = tq.paulis.I()
        E1 = tq.compile(tq.ExpectationValue(U=bra,H=H), backend="qulacs")
        E2 = tq.compile(tq.ExpectationValue(U=ket,H=H), backend="qulacs")

        self.bra = E1.get_expectationvalues()[0]._U
        self.ket = E2.get_expectationvalues()[0]._U
        self.H = E1.get_expectationvalues()[0]._H[0]
        self.n_qubits = ket.n_qubits
        self.is_overlap = H.n_qubits == 0


    def __call__(self, variables, *args, **kwargs):

        self.ket.update_variables(variables)
        self.bra.update_variables(variables)

        state_bra = self.bra.initialize_state(self.n_qubits)
        state_ket = self.ket.initialize_state(self.n_qubits)
        self.bra.circuit.update_quantum_state(state_bra)
        self.ket.circuit.update_quantum_state(state_ket)

        if self.is_overlap:
            vector1 = state_bra.get_vector()
            vector2 = state_ket.get_vector()
            result = vector1.conj().T.dot(vector2)
        else:
            result = self.H.get_transition_amplitude(state_bra, state_ket)

        result=result.real
        return result