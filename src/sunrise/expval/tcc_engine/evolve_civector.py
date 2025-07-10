from tencirchem.static.evolve_civector import *
from tencirchem.static.evolve_civector import _get_gradients_civector,_get_gradients_civector_nocache


def get_expval_and_grad_civector(
    params, hamiltonian, n_qubits, n_elec_s, ex_ops: Tuple, param_ids: Tuple, mode: str = "fermion", init_state=None,
    params_bra=None,ex_ops_bra:Tuple=None,param_ids_bra:Tuple=None,init_state_bra=None):
    # print('opt provided ',params)
    # print('opt provided bra ',params_bra)
    ket = get_civector(params, n_qubits, n_elec_s, ex_ops, param_ids, mode, init_state)
    if ex_ops_bra is None: ex_ops_bra = ex_ops
    if params_bra is None: params_bra = params
    if param_ids_bra is None: param_ids_bra = param_ids
    if init_state_bra is None: init_state_bra = init_state
    params_bra = tc.backend.numpy(params_bra)
    bra = get_civector(params_bra, n_qubits, n_elec_s, ex_ops_bra, param_ids_bra, mode, init_state)
    hbra = tc.backend.numpy(apply_op(hamiltonian, bra))
    hket = tc.backend.numpy(apply_op(hamiltonian, ket))
    energy = hbra @ ket
    # already cached
    op_tensors = get_operator_tensors(n_qubits, n_elec_s, ex_ops, mode=mode)
    op_tensors_bra = get_operator_tensors(n_qubits, n_elec_s, ex_ops_bra, mode=mode)
    theta_tensors = get_theta_tensors(params, param_ids)
    theta_tensors_bra = get_theta_tensors(params_bra, param_ids_bra)
    op_tensors = list(op_tensors) + list(theta_tensors)
    op_tensors_bra = list(op_tensors_bra) + list(theta_tensors_bra)
    gradients_beforesum = _get_gradients_civector(hbra, ket, *op_tensors[1:])
    gradients_beforesum_bra = _get_gradients_civector(hket, bra, *op_tensors_bra[1:])

    gradients_beforesum = tc.backend.numpy(gradients_beforesum)
    gradients = np.zeros(params.shape)
    for grad, param_id in zip(gradients_beforesum, param_ids):
        gradients[param_id] += grad
    gradients_beforesum_bra = tc.backend.numpy(gradients_beforesum_bra)
    gradients_bra = np.zeros(params_bra.shape)
    for grad, param_id in zip(gradients_beforesum_bra, param_ids_bra):
        gradients_bra[param_id] += grad
    print('Gradient ',gradients)
    print('Gradient Bra ',gradients_bra)
    return energy,  gradients+gradients_bra

def get_energy_and_grad_civector(
    params, hamiltonian, n_qubits, n_elec_s, ex_ops: Tuple, param_ids: Tuple, mode: str = "fermion", init_state=None
):
    ket = get_civector(params, n_qubits, n_elec_s, ex_ops, param_ids, mode, init_state)
    bra = apply_op(hamiltonian, ket)
    energy = bra @ ket
    # already cached
    op_tensors = get_operator_tensors(n_qubits, n_elec_s, ex_ops, mode=mode)
    theta_tensors = get_theta_tensors(params, param_ids)
    op_tensors = list(op_tensors) + list(theta_tensors)
    gradients_beforesum = _get_gradients_civector(bra, ket, *op_tensors[1:])

    gradients_beforesum = tc.backend.numpy(gradients_beforesum)
    gradients = np.zeros(params.shape)
    for grad, param_id in zip(gradients_beforesum, param_ids):
        gradients[param_id] += grad

    return energy, 2 * gradients


def get_expval_and_grad_civector_nocache(
    params, hamiltonian, n_qubits, n_elec_s, ex_ops: Tuple, param_ids: Tuple, mode: str = "fermion", init_state=None,
    params_bra=None,ex_ops_bra:Tuple=None,param_ids_bra:Tuple=None,init_state_bra=None):
    ket = get_civector_nocache(params, n_qubits, n_elec_s, ex_ops, param_ids, mode, init_state)
    if ex_ops_bra is None: ex_ops_bra = ex_ops
    if params_bra is None: params_bra = params
    if param_ids_bra is None: param_ids_bra = param_ids
    if init_state_bra is None: init_state_bra = init_state
    bra = get_civector_nocache(params_bra, n_qubits, n_elec_s, ex_ops_bra, param_ids_bra, mode, init_state_bra)
    hbra = apply_op(hamiltonian, bra)
    hket = apply_op(hamiltonian, ket)
    energy = hbra @ ket

    gradients_beforesum = _get_gradients_civector_nocache(hbra, ket, params, n_qubits, n_elec_s, ex_ops, param_ids, mode)
    gradients_beforesum = tc.backend.numpy(gradients_beforesum)
    gradients_beforesum_bra = _get_gradients_civector_nocache(hket, bra, params_bra, n_qubits, n_elec_s, ex_ops_bra, param_ids_bra, mode)
    gradients_beforesum_bra = tc.backend.numpy(gradients_beforesum_bra)

    gradients = np.zeros(params.shape)
    for grad, param_id in zip(gradients_beforesum, param_ids):
        gradients[param_id] += grad
    gradients_bra = np.zeros(params_bra.shape)
    for grad, param_id in zip(gradients_beforesum_bra, param_ids_bra):
        gradients_bra[param_id] += grad
    return energy, gradients+gradients_bra


def get_energy_and_grad_civector_nocache(
    params, hamiltonian, n_qubits, n_elec_s, ex_ops: Tuple, param_ids: Tuple, mode: str = "fermion", init_state=None
):
    ket = get_civector_nocache(params, n_qubits, n_elec_s, ex_ops, param_ids, mode, init_state)
    bra = apply_op(hamiltonian, ket)
    energy = bra @ ket

    gradients_beforesum = _get_gradients_civector_nocache(bra, ket, params, n_qubits, n_elec_s, ex_ops, param_ids, mode)
    gradients_beforesum = tc.backend.numpy(gradients_beforesum)

    gradients = np.zeros(params.shape)
    for grad, param_id in zip(gradients_beforesum, param_ids):
        gradients[param_id] += grad

    return energy, 2 * gradients