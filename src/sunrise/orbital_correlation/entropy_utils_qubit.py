import tequila as tq
from tequila import QCircuit,QubitWaveFunction,QubitHamiltonian
from tequila.objective.objective import Variables
from tequila.quantumchemistry.qc_base import QuantumChemistryBase as tqMolecule
import numpy as np
from scipy.linalg import logm, eigh
import itertools
import math
from typing import List,Tuple,Union

class Input_State:
    """
    Wrapper class to hold the input state to measure the operators
    The input state can be a quantum circuit or a wavefunction
    """
    circuit = None
    variables = None
    wavefunction = None

    def __init__(self, circuit=None, variables=None, wavefunction=None):
        if circuit is None and wavefunction is None:
            raise ValueError("Either a circuit or a wavefunction must be provided")
        if circuit is not None and wavefunction is not None and wavefunction != 0:
            raise ValueError("Only one of circuit or wavefunction must be provided")
        
        self.circuit = circuit
        self.variables = variables
        self.wavefunction = wavefunction

    def get_circuit(self):
        """
        Get the circuit from the wavefunction or the circuit
        """
        if self.wavefunction is not None and self.circuit is None:
            self.circuit = tq.gates.Rz(angle=0, target=range(self.wavefunction.n_qubits)) # dummy empty circuit
        return self.circuit
        
    def get_wavefunction(self, variables=None):
        """
        Get the wavefunction from the circuit or the wavefunction
        """
        if self.circuit is not None and self.wavefunction is None:
            if variables is None:
                variables = self.variables
            self.wavefunction = tq.compile(self.circuit)(variables)
        return self.wavefunction

def compute_one_orb_rdm(mol:tqMolecule, circuit:QCircuit=None, variables:Variables=None, initial_state:QubitWaveFunction=None, one_orb:int=0)->np.ndarray:

    # Initialize the input state in a wrapper class
    state = Input_State(circuit=circuit, variables=variables, wavefunction=initial_state)
    if initial_state is not None and circuit is None:
        circuit = state.get_circuit()
        initial_state = state.get_wavefunction()
    elif circuit is None and initial_state is None:
        raise ValueError("Either a circuit or a wavefunction must be provided")
    up = mol.transformation.up(one_orb)
    down = mol.transformation.down(one_orb)
    ops = {
        "vacuum": tq.paulis.I(),
        "a_pu_dag": mol.make_creation_op(up),
        "a_pu": mol.make_annihilation_op(up),
        "n_pu": mol.make_number_op(up),
        "a_pd_dag": mol.make_creation_op(down),
        "a_pd": mol.make_annihilation_op(down),
        "n_pd": mol.make_number_op(down)
        }
    
    row = [
        ["vacuum"], # |00>
        ["a_pd_dag"], # |0down>
        ["a_pu_dag"], # |up0>
        ["a_pu_dag","a_pd_dag"] # |updown>
    ]

    # Create column as the daggered row
    column = []
    for tmp_ops in row:
        col_ops = []
        for op in tmp_ops:
            if op.endswith("_dag"):
                col_ops.append(op[:-4]) # removes the last 4 characters "_dag"
            else: 
                col_ops.append(op)
        # reverse as we are daggering
        col_ops.reverse()
        column.append(col_ops)

    rho = np.zeros((4,4))
    for i,state_i in enumerate(row):
        for j,state_j in enumerate(column):
            if i==j: # Select only specific terms as in https://iopscience.iop.org/article/10.1088/2058-9565/aca4ee/meta Eq.(27) 
                     # but the order is spin-down first and then spin-up
                
                # for convenience we multiply individual operators before
                op_i = tq.paulis.I()
                for v in state_i:
                    op_i *= ops[v]
                op_j = tq.paulis.I()
                for v in state_j:
                    op_j *= ops[v]

                P = op_i * (1-ops["n_pu"])*(1-ops["n_pd"]) * op_j

                if P.is_hermitian():
                    rho[i][j] = tq.simulate(tq.ExpectationValue(circuit,P), variables=variables, initial_state=initial_state)
                else:
                    P_herm, P_non_herm = P.split()
                    rho[i][j] = tq.simulate(tq.ExpectationValue(circuit,P_herm), variables=variables, initial_state=initial_state) + \
                        tq.simulate(tq.ExpectationValue(circuit,-1j * P_non_herm), variables=variables, initial_state=initial_state)

    return rho

def compute_two_orb_rdm(mol:tqMolecule, circuit:QCircuit=None, variables:Variables=None, initial_state:QubitWaveFunction=None, p_orb:int=0, q_orb:int=1, PSSR:bool=False, NSSR:bool=False)->np.ndarray:

    # Initialize the input state in a wrapper class
    state = Input_State(circuit=circuit, variables=variables, wavefunction=initial_state)
    if initial_state is not None and circuit is None:
        circuit = state.get_circuit()
        initial_state = state.get_wavefunction()
    elif circuit is None and initial_state is None:
        raise ValueError("Either a circuit or a wavefunction must be provided")
    
    pup = mol.transformation.up(p_orb)
    pdown = mol.transformation.down(p_orb)
    qup = mol.transformation.up(q_orb)
    qdown = mol.transformation.down(q_orb)

    ops = {
        "vacuum": tq.paulis.I(),
        "a_pu_dag": mol.make_creation_op(pup),
        "a_pu": mol.make_annihilation_op(pup),
        "n_pu": mol.make_number_op(pup),
        "a_pd_dag": mol.make_creation_op(pdown),
        "a_pd": mol.make_annihilation_op(pdown),
        "n_pd": mol.make_number_op(pdown),
        "a_qu_dag": mol.make_creation_op(qup),
        "a_qu": mol.make_annihilation_op(qup),
        "n_qu": mol.make_number_op(qup),
        "a_qd_dag": mol.make_creation_op(qdown),
        "a_qd": mol.make_annihilation_op(qdown),
        "n_qd": mol.make_number_op(qdown)
    }

    # For row and column order refer to https://pubs.acs.org/doi/10.1021/acs.jctc.0c00559 Figure 4
    row = [
        ["vacuum"], # |0000>
        ["a_qd_dag"], ["a_pd_dag"], # |000down>, |0down00>
        ["a_qu_dag"], ["a_pu_dag"], # |00up0>, |up000>
        ["a_pd_dag","a_qd_dag"], # |0down0down>
        ["a_qu_dag","a_qd_dag"], ["a_pd_dag","a_qu_dag"], ["a_pu_dag","a_qd_dag"], ["a_pu_dag","a_pd_dag"], # |00updown>, |0downup0>, |up00down>, |updown00>
        ["a_pu_dag","a_qu_dag"], # |up0up0>
        ["a_pd_dag","a_qu_dag","a_qd_dag"], ["a_pu_dag","a_pd_dag","a_qd_dag"], # |0downupdown>, |updown0down>
        ["a_pu_dag","a_qu_dag","a_qd_dag"], ["a_pu_dag","a_pd_dag","a_qu_dag"], # |up0updown>, |updownup0>
        ["a_pu_dag","a_pd_dag","a_qu_dag","a_qd_dag"] # |updownupdown>
    ]
    # Create column as the daggered row
    column = []
    for tmp_ops in row:
        col_ops = []
        for op in tmp_ops:
            if op.endswith("_dag"):
                col_ops.append(op[:-4]) # removes the last 4 characters "_dag"
            else: 
                col_ops.append(op)
        # reverse as we are daggering
        col_ops.reverse()
        column.append(col_ops)

    rho = np.zeros((16,16)) # 16x16 matrix
    for i,state_i in enumerate(row):
        for j,state_j in enumerate(column):
            if (i == j or 
                (i in {1, 2} and j in {1, 2}) or
                (i in {3, 4} and j in {3, 4}) or
                (i in {6, 7, 8, 9} and j in {6, 7, 8, 9}) or
                (i in {11, 12} and j in {11, 12}) or
                (i in {13, 14} and j in {13, 14})): # Select only specific terms as in https://pubs.acs.org/doi/10.1021/acs.jctc.0c00559 Figure 4

                # In order to evaluate SSR we make a local number counter for p and q on state i and state j
                N_p_i = N_q_i = N_p_j = N_q_j = 0
                for v,w in zip(state_i,state_j):
                    if v.startswith("a_p"):
                        N_p_i += 1
                    elif v.startswith("a_q"):
                        N_q_i += 1
                    if w.startswith("a_p"):
                        N_p_j += 1
                    elif w.startswith("a_q"):
                        N_q_j += 1
                # P-SSR (odd or even number of particles on local orbitals)
                if PSSR and (N_p_i%2!=N_p_j%2 or N_q_i%2!=N_q_j%2):
                    continue
                # N-SSR (same number of particles on local orbitals)
                if NSSR and (N_p_i!=N_p_j or N_q_i!=N_q_j):
                    continue

                # for convenience we multiply individual operators before
                op_i = tq.paulis.I()
                for v in state_i:
                    op_i *= ops[v]
                op_j = tq.paulis.I()
                for v in state_j:
                    op_j *= ops[v]

                P = op_i * (1-ops["n_pu"])*(1-ops["n_pd"])*(1-ops["n_qu"])*(1-ops["n_qd"]) * op_j
                if P.is_hermitian():
                    rho[i][j] = tq.simulate(tq.ExpectationValue(circuit,P), variables=variables, initial_state=initial_state)
                else:
                    P_herm, P_non_herm = P.split()
                    rho[i][j] = tq.simulate(tq.ExpectationValue(circuit,P_herm), variables=variables, initial_state=initial_state) + \
                        tq.simulate(tq.ExpectationValue(circuit,-1j * P_non_herm), variables=variables, initial_state=initial_state)

    return rho

# Quantum entropy S(rho)
def quantum_entropy(rho:np.ndarray)->float:
    """
    Compute the quantum entropy S(rho).

    Parameters:
        rho (ndarray): Density matrix rho (Hermitian, positive semidefinite, trace = 1).

    Returns:
        float: The quantum entropy S(rho).
    """
    # Ensure rho is a numpy array
    rho = np.array(rho, dtype=np.complex128)

    # Validate the input density matrix
    if not np.allclose(rho, rho.conj().T):
        raise ValueError("rho must be Hermitian.")
    if not np.isclose(np.trace(rho), 1):
        raise ValueError("Trace of rho must be 1.")
    if np.any(np.linalg.eigvalsh(rho).round() < 0):
        raise ValueError("rho must be positive semidefinite.")

    # Compute directly
    # log_rho = logm(rho)
    # entropy = -np.trace(rho @ log_rho).real

    # Compute through eigenvalues
    rho_evals, rho_evecs = eigh(rho)
    rho_evals = np.clip(rho_evals, 1e-12, None)
    log_rho = rho_evecs @ np.diag(np.log(rho_evals)) @ rho_evecs.conj().T
    entropy = -np.trace(rho @ log_rho).real

    return entropy

# Quantum relative entropy S(rho||sigma)
def quantum_relative_entropy(rho:np.ndarray, sigma:np.ndarray)->float:
    """
    Compute the quantum relative entropy S(rho || sigma).

    Parameters:
        rho (ndarray): Density matrix rho (Hermitian, positive semidefinite, trace = 1).
        sigma (ndarray): Density matrix sigma (Hermitian, positive semidefinite, trace = 1).

    Returns:
        float: The quantum relative entropy S(rho || sigma).
    """
    # Ensure rho and sigma are numpy arrays
    rho = np.array(rho, dtype=np.complex128)
    sigma = np.array(sigma, dtype=np.complex128)

    # Validate the input density matrices
    if not np.allclose(rho, rho.conj().T):
        raise ValueError("rho must be Hermitian.")
    if not np.allclose(sigma, sigma.conj().T):
        raise ValueError("sigma must be Hermitian.")
    if not np.isclose(np.trace(rho), 1):
        raise ValueError("Trace of rho must be 1.")
    if not np.isclose(np.trace(sigma), 1):
        raise ValueError("Trace of sigma must be 1.")
    if np.any(np.linalg.eigvalsh(rho).round() < 0):
        raise ValueError("rho must be positive semidefinite.")
    if np.any(np.linalg.eigvalsh(sigma).round() < 0):
        raise ValueError("sigma must be positive semidefinite.")

    # Add some noise to make it positive semidefinite
    # epsilon = 1e-10
    # rho = (rho + epsilon * np.eye(sigma.shape[0])).real
    # sigma = (sigma + epsilon * np.eye(sigma.shape[0])).real

    # Compute directly
    # log_rho = logm(rho)
    # log_sigma = logm(sigma)
    # relative_entropy = np.trace(rho @ (log_rho - log_sigma)).real

    # Compute through eigenvalues
    rho_evals, rho_evecs = eigh(rho)
    sigma_evals, sigma_evecs = eigh(sigma)
    rho_evals = np.clip(rho_evals, 1e-12, None)
    sigma_evals = np.clip(sigma_evals, 1e-12, None)
    log_rho = rho_evecs @ np.diag(np.log(rho_evals)) @ rho_evecs.conj().T
    log_sigma = sigma_evecs @ np.diag(np.log(sigma_evals)) @ sigma_evecs.conj().T
    relative_entropy = np.trace(rho @ (log_rho - log_sigma)).real

    return relative_entropy

# def mutual_info(rho, reduced_rhos):
#     """
#     reduced_rhos is a list of all the reduced density matrices of rho
#     corresponding to the subspaces considered
#     """
#     I = - quantum_entropy(rho)
#     for reduced_rho in reduced_rhos:
#         I += quantum_entropy(reduced_rho)
#     # return I*0.5 # there might be a 0.5 depending to convention
#     return I

def mutual_info_2ordm(mol:tqMolecule, circuit:QCircuit=None, variables:Variables=None, initial_state:QubitWaveFunction=None, orb_a:int=0, orb_b:int=1, PSSR:bool=False, NSSR:bool=False)->float:
    rho_a = compute_one_orb_rdm(mol, circuit, variables, initial_state, orb_a)
    S_a = quantum_entropy(rho_a)
    rho_b = compute_one_orb_rdm(mol, circuit, variables, initial_state, orb_b)
    S_b = quantum_entropy(rho_b)
    rho_ab = compute_two_orb_rdm(mol, circuit, variables, initial_state, p_orb=orb_a, q_orb=orb_b, PSSR=PSSR, NSSR=NSSR)
    S_ab = quantum_entropy(rho_ab)

    # return 0.5 * (S_a + S_b - S_ab) # there might be a 0.5 depending to convention
    return S_a + S_b - S_ab

def mutual_info_1ordm(mol:tqMolecule, circuit:QCircuit=None, variables:Variables=None, initial_state:QubitWaveFunction=None, orb_a:int=0, orb_b:int=1, PSSR:bool=False, NSSR:bool=False): # TODO: orb_b is not necessary because I'm using only orb_a
    rho_a = compute_one_orb_rdm(mol, circuit, variables, initial_state, orb_a)
    rho_b = compute_one_orb_rdm(mol, circuit, variables, initial_state, orb_b)
    if PSSR:
        rho_a_evals, rho_a_evecs = eigh(rho_a)
        I = (rho_a_evals[0]+rho_a_evals[3])*np.log(rho_a_evals[0]+rho_a_evals[3]) + \
            (rho_a_evals[1]+rho_a_evals[2])*np.log(rho_a_evals[1]+rho_a_evals[2]) - \
            2*(rho_a_evals[0]*np.log(rho_a_evals[0])+rho_a_evals[1]*np.log(rho_a_evals[1])+\
               rho_a_evals[2]*np.log(rho_a_evals[2])+rho_a_evals[3]*np.log(rho_a_evals[3]))
    elif NSSR:
        rho_a_evals, rho_a_evecs = eigh(rho_a)
        I = rho_a_evals[0]*np.log(rho_a_evals[0]) + \
            (rho_a_evals[1]+rho_a_evals[2])*np.log(rho_a_evals[1]+rho_a_evals[2]) + \
            rho_a_evals[3]*np.log(rho_a_evals[3]) - \
            2*(rho_a_evals[0]*np.log(rho_a_evals[0])+rho_a_evals[1]*np.log(rho_a_evals[1])+\
               rho_a_evals[2]*np.log(rho_a_evals[2])+rho_a_evals[3]*np.log(rho_a_evals[3]))
    else:
        # S_a = quantum_entropy(rho_a)
        # S_b = quantum_entropy(rho_b)
        # rho_ab = compute_two_orb_rdm(mol, circuit, p_orb=orb_a, q_orb=orb_b, PSSR=PSSR, NSSR=NSSR)
        # S_ab = quantum_entropy(rho_ab)
        # I = S_a + S_b - S_ab
        I = 2*pure_state_entanglement(mol, circuit, variables, initial_state, orb_a=orb_a, orb_b=orb_b)

    return I

def total_mutual_info(mol, circuit=None, variables=None, initial_state=0, orbs=[0,1], PSSR=False, NSSR=False):
    """
    Compute the mutual information between specified orbitals and the rest of the system.
    """
    one_entropies = []
    for orb in orbs:
        one_rdm = compute_one_orb_rdm(mol, circuit, variables, initial_state, orb)
        one_entropies.append(quantum_entropy(one_rdm))
    one_entropy = sum(one_entropies)

    state = tq.simulate(circuit, variables=variables, initial_state=initial_state)
    rho = tq.paulis.Projector(state).to_matrix().real
    system_entropy = quantum_entropy(rho)

    return one_entropy - system_entropy

def pure_state_entanglement(mol:tqMolecule, circuit:QCircuit=None, variables:Variables=None, initial_state:QubitWaveFunction=None, orb_a:int=0, orb_b:int=1, PSSR:bool=False, NSSR:bool=False)->float:
    rho_a = compute_one_orb_rdm(mol, circuit, variables, initial_state, orb_a)
    rho_b = compute_one_orb_rdm(mol, circuit, variables, initial_state, orb_b)
    if PSSR==True:
        rho_a_evals, rho_a_evecs = eigh(rho_a)
        # Eq.(29) https://doi.org/10.1021/acs.jctc.0c00559
        plnp = [(rho_a_evals[i]+rho_a_evals[-1-i])*np.log(rho_a_evals[i]+rho_a_evals[-1-i]) if not np.isclose(rho_a_evals[i]+rho_a_evals[-1-i], 0, atol=1.e-6) else 0.0 for i in range(len(rho_a_evals)//2)]
        xlnx =[i * np.log(i) if not np.isclose(i, 0, atol=1.e-6) else 0.0 for i in rho_a_evals]
        E = (plnp[0]+plnp[1] ) - (xlnx[0]+xlnx[1]+xlnx[2]+xlnx[3])
    elif NSSR==True:
        rho_a_evals, rho_a_evecs = eigh(rho_a)
        # Eq.(29) https://doi.org/10.1021/acs.jctc.0c00559
        xlnx =[i * np.log(i) if not np.isclose(i, 0, atol=1.e-6) else 0.0 for i in rho_a_evals]
        E = (rho_a_evals[1]+rho_a_evals[2])*np.log(rho_a_evals[1]+rho_a_evals[2]) if not np.isclose(rho_a_evals[1]+rho_a_evals[2], 0, atol=1.e-6) else 0.0
        E -= (xlnx[1] + xlnx[2])
    else:
        S_a = quantum_entropy(rho_a)
        S_b = quantum_entropy(rho_b)
        # assert np.isclose(S_a,S_b)
        E = S_a

    return E

# def full_trace_projectors(projector:QubitHamiltonian, n_qubits:int=None)->list:

#     # n_qubits refers to the number of qubits of the reduced density matrix
#     zero = tq.QubitWaveFunction.from_string('|0>')
#     one = tq.QubitWaveFunction.from_string('|1>')
#     if n_qubits:
#         states = list(itertools.product([zero, one],repeat=n_qubits))
#         qubits = list(range(1,n_qubits+2))
#         traced_qubits = list(range(1,n_qubits+2))
#     else:
#         states = list(itertools.product([zero, one],repeat=projector.n_qubits-1))
#         qubits = projector.qubits
#         traced_qubits = projector.qubits

#     traced_pjs = []
#     for qubit in qubits:
#         traced_qubits.remove(qubit)

#         temp = []
#         for state in states:
#             temp.append(projector.trace_out_qubits(qubits=traced_qubits, states=state))

#         temp = sum(temp)
#         traced_pjs.append(temp)
#         if n_qubits:
#             traced_qubits = list(range(1,n_qubits+2))
#         else:
#             traced_qubits = projector.qubits
#     return traced_pjs

# def part_trace_projectors(projector:QubitHamiltonian, reduced_qubits_list:List[int], projector_qubits:bool=None)->list:
#     """
#     reduced_qubits_list:
#         is a list of lists with the qubits that we want to keep in our reduced density matrix,
#         if we have qubits [0,1,2,3] and we want to trace out [2,3] first and [0] second
#         then reduced_qubits_list will be: [[0,1],[1,2,3]]

#     projector_qubits:
#         is the list of qubits on which the projector applies to,
#         this is needed only if the projector is a single I() operator,
#         for example a 4 qubit system will have: [0,1,2,3]
#     """
#     zero = tq.QubitWaveFunction.from_string('|0>')
#     one = tq.QubitWaveFunction.from_string('|1>')

#     if not projector_qubits:
#         projector_qubits = projector.qubits

#     traced_pjs = []
#     for reduced_qubits in reduced_qubits_list:
#         assert len(reduced_qubits) <= len(projector_qubits), "reduced_qubits must be less or equal than projector_qubits"

#         traced_qubits = [q for q in projector_qubits if q not in reduced_qubits]

#         states = list(itertools.product([zero, one], repeat=len(traced_qubits)))

#         temp = []
#         for state in states:
#             temp.append(projector.trace_out_qubits(qubits=traced_qubits, states=state))
        
#         temp = sum(temp)
#         traced_pjs.append(temp)

#     if len(traced_pjs)==1:
#         traced_pjs = traced_pjs[0]
        
#     return traced_pjs

# def np_density_matrix(rho:Union[np.ndarray,QubitHamiltonian], n_qubits:int=None)->np.ndarray:

#     if isinstance(rho,np.ndarray):
#         rho_matrix = rho
#         return rho_matrix

#     try:
#         rho_matrix = rho.to_matrix().real
#     except:
#         # in case the operator is tq.paulis.I() tequila cannot cast into a matrix
#         rho_matrix = list(rho.values())[0].real * np.eye(2**(n_qubits))

#     return rho_matrix

# def product_state_density(n_qubits:int)->QubitHamiltonian:

#     sigma = 0.5 * tq.paulis.I()
#     for i in range(1,n_qubits):
#         sigma = np.kron(sigma, 0.5 * tq.paulis.I())

#     return sigma

# def pauli_decomposition(matrix:np.ndarray)->Tuple[QubitHamiltonian,QubitHamiltonian]:
#     n_qubits = np.ceil(np.log2(max(matrix.shape)))
#     n_qubits = int(n_qubits)

#     H = tq.paulis.Zero()
#     for i in range(matrix.shape[0]):
#         for j in range(matrix.shape[1]):
#             H += matrix[i,j] * tq.paulis.KetBra(ket=i, bra=j, n_qubits=n_qubits)

#     hermitian, anti = H.split()
#     return hermitian, anti
