from tencirchem.static.evolve_pyscf import *
from tencirchem.static.evolve_pyscf import _get_gradients_pyscf


def get_expval_and_grad_pyscf(
    params, hamiltonian, n_qubits, n_elec_s, ex_ops: Tuple, param_ids: Tuple, mode: str = "fermion", init_state=None,
    params_bra=None,ex_ops_bra:Tuple=None,param_ids_bra:Tuple=None,init_state_bra=None):
    params = tc.backend.numpy(params)
    ket = get_civector_pyscf(params, n_qubits, n_elec_s, ex_ops, param_ids, mode, init_state)
    if ex_ops_bra is None: ex_ops_bra = ex_ops
    if params_bra is None: params_bra = params
    if param_ids_bra is None: param_ids_bra = param_ids
    if init_state_bra is None: init_state_bra = init_state
    params_bra = tc.backend.numpy(params_bra)
    bra = get_civector_pyscf(params_bra, n_qubits, n_elec_s, ex_ops_bra, param_ids_bra, mode, init_state_bra) 
    hbra = tc.backend.numpy(apply_op(hamiltonian, bra))
    hket = tc.backend.numpy(apply_op(hamiltonian, ket))
    energy = hbra @ ket

    gradients_beforesum = _get_gradients_pyscf(bra=hbra, ket=ket, params=params, n_qubits=n_qubits, n_elec_s=n_elec_s, ex_ops=ex_ops, param_ids=param_ids, mode=mode)
    gradients_beforesum_bra = _get_gradients_pyscf(ket=bra, bra=hket, params=params_bra, n_qubits=n_qubits, n_elec_s=n_qubits, ex_ops=ex_ops_bra, param_ids=param_ids_bra, mode=mode)

    gradients = np.zeros(params.shape)
    gradients_bra = np.zeros(params_bra.shape)
    for grad, param_id in zip(gradients_beforesum, param_ids):
        gradients[param_id] += grad
    for grad, param_id in zip(gradients_beforesum_bra, param_ids_bra):
        gradients_bra[param_id] += grad
    return energy, gradients + gradients_bra


def get_energy_and_grad_pyscf(
    params, hamiltonian, n_qubits, n_elec_s, ex_ops: Tuple, param_ids: Tuple, mode: str = "fermion", init_state=None
):
    params = tc.backend.numpy(params)
    ket = get_civector_pyscf(params, n_qubits, n_elec_s, ex_ops, param_ids, mode, init_state)
    bra = tc.backend.numpy(apply_op(hamiltonian, ket))
    energy = bra @ ket

    gradients_beforesum = _get_gradients_pyscf(bra, ket, params, n_qubits, n_elec_s, ex_ops, param_ids, mode)

    gradients = np.zeros(params.shape)
    for grad, param_id in zip(gradients_beforesum, param_ids):
        gradients[param_id] += grad

    return energy, 2 * gradients