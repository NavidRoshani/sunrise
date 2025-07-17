import tequila as tq
import sunrise as sun
import numpy as np
from numbers import Number
from scipy.optimize import minimize
import tencirchem as tcc
from tencirchem.utils.optimizer import soap
from tequila.utils.bitstrings import BitString, reverse_int_bits
from sunrise.expval.tcc_engine.braket import EXPVAL
from sunrise.expval.pyscf_molecule import from_tequila
from copy import deepcopy
# # print(tq.QubitWaveFunction.from_string('|11110000>'))
# #
# # exit()
# # from tencirchem import UCC,UCCSD
# #
# # from tencirchem.molecule import h2
# #
# # ucc = UCCSD(h2)
# # ucc.kernel()
# # print(ucc.init_state)
# # print(ucc.civector([0, 0]))
# # print(ucc.init_state)
# # print(ucc.civector())
# # exit()
# # a = []
# # a.append('b')
# # a.insert(0,"a")
# # print(a)
# # a = "hola"
# # for i,j in enumerate(a):
# #     print(f'{i}-->{j}')
# # exit()
# geo = "H 0. 0. 0. \n H 0. 0. 1.\n H 0. 0. 2. \n H 0. 0. 3."
# mol = tq.Molecule(geometry=geo, basis_set="sto-3g")
# fci = mol.compute_energy(method="fci")
# print('FCI ',fci)
# hf = mol.compute_energy(method="hf")
# print("HF ",hf)
# mol = mol.use_native_orbitals()
# U = mol.make_ansatz('HCB-SPA',edges=[(0,1),(2,3)])
# init_guess = np.eye(mol.n_orbitals)
# init_guess[0,0] = 1.
# init_guess[1,0] = 1.
# init_guess[0,1] = -1.
# init_guess[1,1] = 1.
# init_guess[2,2] = 1.
# init_guess[3,2] = 1.
# init_guess[2,3] = -1.
# init_guess[3,3] = 1.
# opt = tq.quantumchemistry.optimize_orbitals(molecule=mol,circuit=U,initial_guess=init_guess,silent=True,use_hcb=True)
# H = opt.molecule.make_hamiltonian()
# SPA = tq.minimize(tq.ExpectationValue(H=H,U=U+mol.hcb_to_me()),silent=True)
# print("SPA E: ",SPA.energy)
#
# # h = opt.molecule.integral_manager.one_body_integrals
# # g = opt.molecule.integral_manager.two_body_integrals
# # c = opt.molecule.integral_manager.constant_term
# c,h,g = mol.get_integrals()
# g = g.reorder('chem')
# g = g.elems
# s = opt.molecule.integral_manager.overlap_integrals
# # h = mol.integral_manager.one_body_integrals
# # g = mol.integral_manager.two_body_integrals
# # c = mol.integral_manager.constant_term
# # g = g.reorder('chem')
# # g = g.elems
# # IT COULD BE ALSO USED THE MOL.PYSCF_MOLECULE
# U = tcc.UCC.from_integral(int1e=h, int2e=g,n_elec= mol.n_electrons, e_core=c,ovlp=s,**{"mo_coeff": opt.mo_coeff, "init_method": "zeros", "run_hf": True, "run_mp2": False, "run_ccsd": False, "run_fci": True})#.kernel()
# print("New ",U.e_fci)
# # print(U.get_init_state_dataframe())
# # U.print_ansatz()
# # print(U.ex_ops)
# # print(U.params)
# # print(U.param_to_ex_ops)
# # init_state = [0]*len(U.init_state)
#
# # U.init_state = [1,1,0,0,1,1,0,0]
# # print(U.get_init_state_dataframe())
# # exit()
# # print(U.statevector())
# # print(U.params)
# # print(U.param_ids)
# # print(U.param_to_ex_ops)
# # print(U.energy([0,0]))
# # print(U.e_fci)
# # # tcc.UCC.get
# # U.ex_ops = []
# # U.ex_ops = [(0,2,1,3),(4,6,5,7)]
# # U.params = ['a','b']
# # U.param_ids = [0,1]
#
# exp_ops, params,param_ids = from_indices([((0,2),(1,3)),((4,6),(5,7))],mol.n_orbitals) #
# U.ex_ops = exp_ops
# U.params = params
# U.param_ids = param_ids
# # init_state = reference('11110000',mol.n_orbitals)
# ref =  tq.QubitWaveFunction.from_string('|11001100>').to_array()[::-1]
# # print("ref ",ref)
# U.civector_fci = ref
# # print("init_state ",init_state)
# print('ex_ops ',U.ex_ops)
# print('params',U.params)
# print('param_ids ',U.param_ids)
# print(U.get_init_state_dataframe())
#
# U.kernel()
# print('kernel ',U.kernel())
# var = SPA.variables
#
# # print('E with spa angle',U.energy([0.35241897500816977]))
# print('Ref energy',U.energy([0]*len(param_ids)))
# # U.print_summary()
# # U.param_ids = None # decoupled singles
# # print(U.param_ids)
# # print(U.param_to_ex_ops)
# # print(U.t1)
# # print(U.t2)
# # circuit = U.get_circuit()
# # # cost: input parameters return energy(params)
# #
# # U.get_ex1_ops()
# wv = tq.QubitWaveFunction.from_string('0.707|11110000>+0.707|11100100>').normalize()
# print(wv)
# print(vars(wv))
# wv.numbering = 1
# print(vars(wv))
# print(wv._state)
# circuit = [[(0,2),(1,3)],]
# nq = max([max([max([idx[0] for idx in exct]+[idx[1] for idx in exct])]) for exct in circuit])
# print(nq)
# U = tq.QCircuit().to_networkx()
# import tensorcircuit as tc
# U = tq.paulis.X([0,1,2,3]).to_matrix()
# print(U.to_matrix())
# U = tc.Circuit()
# U = tc.circuit.expectation()

def from_edges(edges,nmo):
    '''
    Expected edges like [(0,1),(2,3),...]
    Returned [(0,0+nno,...,1+nmo,2),(a,c,...,d,b)]
    '''

    assert isinstance(edges,(list,tuple))
    if isinstance(edges[0],Number):
        edges = [edges]
    indices = []
    for edge in edges:
        if len(edge) == 1:
            print(f"Include the edge {edge} on the reference state.")
        ext = []
        for i in range(len(edge)-1):
            ext.append((2*edge[i],2*edge[i+1]))
            ext.append((2*edge[i]+1,2*edge[i+1]+1))
        indices.append(ext)
    print('Indices ',indices )
    ex_ops = []
    params = []
    param_ids = []
    for exct in indices:
        exc = []
        params.append(str(exct))
        param_ids.append(len(param_ids))
        for idx in exct:
            exc.append(idx[0]//2+(idx[0]%2)*nmo)
            exc.insert(0,idx[1]//2+(idx[1]%2)*nmo)
        ex_ops.append(tuple(exc))
    return ex_ops,params,param_ids
def from_indices(indices,nmo):
    '''
    Expected indices like [[(0,2),(1,3),...],[(a,b),(c,d),...],...]
    Returned [(0,0+nno,...,1+nmo,2),(a,c,...,d,b)]
    '''
    assert isinstance(indices,(list,tuple))
    if isinstance(indices[0],Number):
        indices = [indices]
    ex_ops = []
    params = []
    param_ids = []
    for exct in indices:
        exc = []
        params.append(str(exct))
        param_ids.append(len(param_ids))
        for idx in exct:
            exc.append(idx[0]//2+(idx[0]%2)*nmo)
            exc.insert(0,idx[1]//2+(idx[1]%2)*nmo)
        ex_ops.append(tuple(exc))
    return ex_ops,params,param_ids
def from_wavefunction(wvf):
    init_state = []
    for i in wvf._state:
        # print(bin(i),wvf._state[i])
        vec = bin(i)[2:]
        vup = ''
        vdw = ''
        for j in range(len(vec)//2):
            vup += vec[2*j]
            vdw += vec[2*j+1]
        vec = (vup + vdw)[::-1]
        init_state.append([vec,wvf._state[i]])
    return init_state
def from_simulated(wvf,tol=1e-6):
    init_state = []
    nq = wvf.n_qubits
    nq = int(2* (np.ceil(nq/2))) #if HCB may be odd amount of qubits
    for i,idx in enumerate(wvf._state):
        vec = BitString.from_int(i)
        vec.nbits = nq
        vec = vec.binary
        vup = ''
        vdw = ''
        for j in range(len(vec)//2):
            vup += vec[2*j]
            vdw += vec[2*j+1]
        vec = (vup + vdw)
        if idx > tol:
            if idx.imag > tol: raise tq.TequilaException(f'TCC only support real wvf coefficients, received: {idx}')
            init_state.append([vec,idx.real])
    return init_state

def prueba_h2():
    print("__________H2_________")
    geo = "H 0. 0. 0. \n H 0. 0. 1."
    mol = tq.Molecule(geometry=geo, basis_set="sto-3g")
    fci = mol.compute_energy(method="fci")
    print('FCI ', fci)
    hf = mol.compute_energy(method="hf")
    print("HF ", hf)
    mol = mol.use_native_orbitals()
    U = mol.make_ansatz('HCB-SPA', edges=[(0, 1)])
    init_guess = np.eye(mol.n_orbitals)
    init_guess[0, 0] = 1.
    init_guess[1, 0] = 1.
    init_guess[0, 1] = -1.
    init_guess[1, 1] = 1.
    opt = tq.quantumchemistry.optimize_orbitals(molecule=mol, circuit=U, initial_guess=init_guess, silent=True,use_hcb=True)
    H = opt.molecule.make_hamiltonian()
    SPA = tq.minimize(tq.ExpectationValue(H=H, U=U + mol.hcb_to_me()), silent=True)
    print("SPA E: ", SPA.energy)
    c, h, g = mol.get_integrals()
    g = g.reorder('chem')
    g = g.elems
    s = opt.molecule.integral_manager.overlap_integrals
    U = tcc.UCC.from_integral(int1e=h, int2e=g,n_elec= mol.n_electrons, e_core=c,ovlp=s,**{"mo_coeff": opt.mo_coeff, "init_method": "zeros", "run_hf": True, "run_mp2": False, "run_ccsd": False, "run_fci": False})
    exp_ops, params, param_ids = from_indices([((0, 2), (1, 3))], mol.n_orbitals)
    U.ex_ops = exp_ops
    U.params = params
    U.param_ids = param_ids
    U.civector_fci = tq.QubitWaveFunction.from_string('|1100>').to_array()[::-1]
    print('ex_ops ', U.ex_ops)
    print('params', U.params)
    print('param_ids ', U.param_ids)
    print(U.get_init_state_dataframe())
    U.kernel()
    print('kernel ', U.kernel())
    print('Ref energy', U.energy([0] * len(param_ids)))


def prueba_h4():
    print("__________H4_________")
    geo = "H 0. 0. 0. \n H 0. 0. 1.\nH 0. 0. 2.\n H 0. 0. 3."
    mol = tq.Molecule(geometry=geo, basis_set="sto-3g")
    fci = mol.compute_energy(method="fci")
    print('FCI ', fci)
    hf = mol.compute_energy(method="hf")
    print("HF ", hf)
    mol = mol.use_native_orbitals()
    U = mol.make_ansatz('HCB-SPA', edges=[(0, 1), (2, 3)])
    init_guess = np.eye(mol.n_orbitals)
    init_guess[0, 0] = 1.
    init_guess[1, 0] = 1.
    init_guess[0, 1] = -1.
    init_guess[1, 1] = 1.
    init_guess[2, 2] = 1.
    init_guess[3, 2] = 1.
    init_guess[2, 3] = -1.
    init_guess[3, 3] = 1.
    opt = tq.quantumchemistry.optimize_orbitals(molecule=mol, circuit=U, initial_guess=init_guess, silent=True,
                                                use_hcb=True)
    H = opt.molecule.make_hamiltonian()
    SPA = tq.minimize(tq.ExpectationValue(H=H, U=U + mol.hcb_to_me()), silent=True)
    print("SPA E: ", SPA.energy)
    print('With variables: ',SPA.variables)
    c, h, g = mol.get_integrals()
    g = g.reorder('chem')
    g = g.elems
    s = opt.molecule.integral_manager.overlap_integrals
    U = tcc.UCC.from_integral(int1e=h, int2e=g, n_elec=mol.n_electrons, e_core=c, ovlp=s,
                              **{"mo_coeff": opt.mo_coeff, "init_method": "zeros", "run_hf": True, "run_mp2": False,
                                 "run_ccsd": False, "run_fci": False})
    exp_ops, params, param_ids = from_indices([((0, 2), (1, 3)), ((4, 6), (5, 7))], mol.n_orbitals)
    init_guess = from_wavefunction(tq.QubitWaveFunction.from_string('|11001100>').normalize())
    # print(init_guess.to_array())
    for i in init_guess:
        i[0] = U.get_addr(i[0])
    init = [0] * U.civector_size
    for i in init_guess:
        init[i[0]] = i[1]
    U.init_state = init
    # print(U.civector_size)
    # exit()
    # print(U.get_init_state_dataframe())
    U.ex_ops = exp_ops
    U.params = params
    U.param_ids = param_ids
    print('ex_ops ', U.ex_ops)
    print('params', U.params)
    print('param_ids ', U.param_ids)
    print('Init variables ',U.init_guess)
    print('kernel ', U.kernel())
    U.kernel()
    print('Ref energy', U.energy([0] * len(param_ids)))
    print('Variables ',U.init_guess)
    # opt_res = minimize(U.energy, U.init_guess, method=soap)
    # print('Optimized ',opt_res.fun)
    # print('Opt Vars ',[2*i for i in opt_res.x])

def prueba_H4_braket():
    print("__________H4_________")
    geo = "H 0. 0. 0. \n H 0. 0. 1.\nH 0. 0. 2.\n H 0. 0. 3."
    mol = tq.Molecule(geometry=geo, basis_set="sto-3g")
    fci = mol.compute_energy(method="fci")
    print('FCI ', fci)
    hf = mol.compute_energy(method="hf")
    print("HF ", hf)
    mol = mol.use_native_orbitals()
    U = mol.make_ansatz('HCB-SPA', edges=[(0, 1), (2, 3)])
    init_guess = np.eye(mol.n_orbitals)
    init_guess[0, 0] = 1.
    init_guess[1, 0] = 1.
    init_guess[0, 1] = -1.
    init_guess[1, 1] = 1.
    init_guess[2, 2] = 1.
    init_guess[3, 2] = 1.
    init_guess[2, 3] = -1.
    init_guess[3, 3] = 1.
    opt = tq.quantumchemistry.optimize_orbitals(molecule=mol, circuit=U, initial_guess=init_guess, silent=True,
                                                use_hcb=True)
    H = opt.molecule.make_hamiltonian()
    SPA = tq.minimize(tq.ExpectationValue(H=H, U=U + mol.hcb_to_me()), silent=True)
    print("SPA E: ", SPA.energy)
    print('With variables: ',SPA.variables)
    c, h, g = mol.get_integrals()
    g = g.reorder('chem')
    g = g.elems
    # print('Given ',g)
    s = opt.molecule.integral_manager.overlap_integrals
    U = tcc.UCC.from_integral(int1e=h, int2e=g, n_elec=mol.n_electrons, e_core=c, ovlp=s,engine='civector',
                              **{"mo_coeff": opt.mo_coeff, "init_method": "zeros", "run_hf": True, "run_mp2": False,
                                 "run_ccsd": False, "run_fci": False})
    exp_ops, params, param_ids = from_indices([((0, 2), (1, 3)), ((4, 6), (5, 7))], mol.n_orbitals)
    init_guess = from_wavefunction(tq.QubitWaveFunction.from_string('|11001100>').normalize())
    # print(init_guess.to_array())
    for i in init_guess:
        i[0] = U.get_addr(i[0])
    init = [0] * U.civector_size
    for i in init_guess:
        init[i[0]] = i[1]
    U.init_state = init
    # print(U.civector_size)
    # exit()
    # print(U.get_init_state_dataframe())
    U.ex_ops = exp_ops
    U.params = params
    U.param_ids = param_ids
    print('ex_ops ', U.ex_ops)
    print('params', U.params)
    print('param_ids ', U.param_ids)
    print('Init variables ',U.init_guess)
    print('Ci Vector ',U.get_ci_strings(),'->',len(U.get_ci_strings()))
    # print('kernel ', U.kernel())
    U.kernel()
    print('Ref energy', U.energy([0] * len(param_ids)))
    psi = U.civector()
    # U.apply_excitation()
    bra = U.civector()
    print(bra.conj() @ psi)
    print(type(bra.conj() @ psi))
    # print(mol.make_hamiltonian().to_matrix())
    # print('->',mol.make_hamiltonian().to_matrix().shape)
    # U.make_rdm1

def prueba_HLi_braket():
    # geo = "H 0. 0. 0. \n H 0. 0. 1.\nH 0. 0. 2.\n H 0. 0. 3."
    geo = "H 0. 0. 0. \n Li 0. 0. 1."
    mol = tq.Molecule(geometry=geo, basis_set="sto-3g")
    H = mol.make_molecular_hamiltonian()
    c,h,g = mol.get_integrals()
    # print(h.shape)
    # c,h,g = mol.integral_manager.get_integrals()
    # print(h.shape)
    # exit()
    hmol = sun.Molecule(geometry=geo, basis_set="sto-3g")
    fci = mol.compute_energy(method="fci")
    print('FCI ', fci)
    hf = mol.compute_energy(method="hf")
    print("HF ", hf)
    mol = mol.use_native_orbitals()
    edges = hmol.get_spa_edges()
    U = mol.make_ansatz('HCB-SPA', edges=edges)
    # w = tq.simulate(U,variables={va:0. for va in U.extract_variables()})
    # print(w)
    # wfn = tq.QubitWaveFunction.from_array(w._state)
    # init_state = tq.QubitWaveFunction.from_string('|10000000>').normalize()
    # print('tq.simulate ',w._state)
    # print('Qwfn ',init_state._state)
    # print('Qwfn from array',wfn._state)
    # print(type(wfn._state))
    # print(type(w._state))
    # print(type(init_state._state))
    # # print(from_wavefunction(init_state))
    # # print(from_simulated(w))
    # print(w)
    # exit()
    init_guess = hmol.get_spa_guess()
    opt = tq.chemistry.optimize_orbitals(molecule=mol, circuit=U, initial_guess=init_guess.T, silent=True,use_hcb=True)
    print('Opt ',opt.energy)
    bra = mol.make_ansatz('SPA', edges=edges)
    # ket = mol.make_ansatz('SPA', edges=[(0, 3), (1, 2)])
    H = opt.molecule.make_hamiltonian()
    print('Tq Hamiltonian: ',H.to_matrix().shape)
    braket = tq.ExpectationValue(U=bra,H=H)
    # braket = tq.BraKet(bra=bra,ket=bra,operator=H)
    # res = tq.minimize(braket[0],silent=True)
    res = tq.minimize(braket,silent=True)
    print(tq.simulate(bra,variables=res.variables))    
    # print(tq.simulate(ket,variables=res.variables))    
    print("Energy ",res.energy)
    print("Variables ",res.variables)
    c, h, g = mol.get_integrals()
    g = g.reorder('chem')
    g = g.elems
    s = opt.molecule.integral_manager.overlap_integrals
    # pyscfmol = tq.chemistry.QuantumChemistryPySCF.from_tequila(molecule=opt.molecule).pyscf_molecule
    # U = tcc.UCC(mol=pyscfmol,* {"mo_coeff": opt.mo_coeff, "init_method": "zeros", "run_hf": True, "run_mp2": False,"run_ccsd": False, "run_fci": False})
    U = tcc.UCC.from_integral(int1e=h, int2e=g, n_elec=mol.n_electrons, e_core=c,**{"mo_coeff": opt.mo_coeff, "init_method": "zeros", "run_hf": True, "run_mp2": False,"run_ccsd": False, "run_fci": False})
    ex_ops, params, param_ids = from_edges(edges=edges, nmo=mol.n_orbitals)
    # init_state = from_wavefunction(tq.QubitWaveFunction.from_string('|11001100>').normalize())
    # print({va:0. for va in bra.extract_variables()})
    init_state = from_simulated(tq.simulate(bra,variables={va:0. for va in bra.extract_variables()}))
    print('F ',init_state)
    # print(init_guess.to_array())
    for i in init_state:
        i[0] = U.get_addr(i[0])
    init = [0] * U.civector_size
    for i in init_state:
        init[i[0]] = i[1]
    U.init_state = init
    U.ex_ops = ex_ops
    U.params = params
    print('init ',init)
    print('ex_ops ',ex_ops)
    print('params ',params)
    print('param_ids ',param_ids)
    U.kernel()
    print(U.opt_res.e)
    print(U.opt_res.x)
    print([2*i for i in U.opt_res.x])
    # print(len(U.aslst))
    print('--------------')
    BK = EXPVAL.from_integral(int1e=h, int2e=g, n_elec=mol.n_electrons, e_core=c,**{"mo_coeff": opt.mo_coeff, "init_method": "zeros", "run_hf": True, "run_mp2": False,"run_ccsd": False, "run_fci": False})
    BK.init_state_bra = init
    BK.init_state = init
    BK.ex_ops_bra = ex_ops
    BK.ex_ops = ex_ops
    BK.params_bra = params
    BK.params = params
    BK.param_ids_bra = param_ids
    BK.param_ids = param_ids
    BK.kernel()
    print(BK.opt_res.e)
    print(BK.opt_res.x)
    print([2*i for i in BK.opt_res.x])

def prueba_Variable():
    geo = "H 0. 0. 0. \n H 0. 0. 1.\nH 0. 0. 2.\n H 0. 0. 3."
    # geo = "H 0. 0. 0. \n Li 0. 0. 1."
    mol = tq.Molecule(geometry=geo, basis_set="sto-3g")
    H = mol.make_molecular_hamiltonian()
    c,h,g = mol.get_integrals()
    hmol = sun.Molecule(geometry=geo, basis_set="sto-3g")
    fci = mol.compute_energy(method="fci")
    print('FCI ', fci)
    hf = mol.compute_energy(method="hf")
    print("HF ", hf)
    mol = mol.use_native_orbitals()
    edges = hmol.get_spa_edges()
    U = mol.make_ansatz('HCB-SPA', edges=edges)
    init_guess = hmol.get_spa_guess()
    opt = tq.chemistry.optimize_orbitals(molecule=mol, circuit=U, initial_guess=init_guess.T, silent=True,use_hcb=True)
    print('Opt ',opt.energy)
    bra = mol.make_ansatz('SPA', edges=edges)
    H = opt.molecule.make_hamiltonian()
    print('Tq Hamiltonian: ',H.to_matrix().shape)
    braket = tq.ExpectationValue(U=bra,H=H)
    res = tq.minimize(braket,silent=True)
    print(tq.simulate(bra,variables=res.variables))    
    print("Energy ",res.energy)
    print("Variables ",res.variables)
    c, h, g = mol.get_integrals()
    g = g.reorder('chem')
    g = g.elems
    U = tcc.UCC.from_integral(int1e=h, int2e=g, n_elec=mol.n_electrons, e_core=c,**{"mo_coeff": opt.mo_coeff, "init_method": "zeros", "run_hf": True, "run_mp2": False,"run_ccsd": False, "run_fci": False})
    ex_ops, params, param_ids = from_edges(edges=edges, nmo=mol.n_orbitals)
    init_state = from_simulated(tq.simulate(bra,variables={va:0. for va in bra.extract_variables()}))
    print('F ',init_state)
    for i in init_state:
        i[0] = U.get_addr(i[0])
    init = [0] * U.civector_size
    for i in init_state:
        init[i[0]] = i[1]
    # U.init_state = init
    # U.ex_ops = ex_ops
    print('init ',init)
    print('ex_ops ',ex_ops)
    a = tq.Variable('b')
    ce = tq.Variable('d')
    # params = tq.Objective([a,2*a])
    params= [a,a]
    param_ids=[0,1]
    # U.params = params
    # U.param_ids = param_ids
    # print('params ',params)
    # print('param_ids ',param_ids)
    # U.kernel()
    # print(U.opt_res.e)
    # print(U.opt_res.x)
    # print([2*i for i in U.opt_res.x])
    # print('--------------')
    # exit()
    BK = EXPVAL.from_integral(int1e=h, int2e=g, n_elec=mol.n_electrons, e_core=c,**{"mo_coeff": opt.mo_coeff, "init_method": "zeros", "run_hf": True, "run_mp2": False,"run_ccsd": False, "run_fci": False})
    BK.init_state_bra = init
    BK.init_state = init
    BK.ex_ops_bra = ex_ops
    BK.ex_ops = ex_ops
    # BK.params = params
    BK.params_bra = params
    BK.params_ket = params
    # BK.param_ids_bra = param_ids
    # BK.param_ids = param_ids
    print('params_bra ',BK.params_bra)
    print('params_ket ',BK.params_ket)
    print('init_state_ket ',BK.init_state_ket)
    print('init_state_brat ',BK.init_state_bra)
    print('variables_bra ',[i.extract_variables() for i in BK.variables_bra])
    print('variables_ket ',BK.variables_ket)
    print('param_to_var_bra ',BK.param_to_var_bra)
    print('param_to_var_ket ',BK.param_to_var_ket)
    print("N variables Bra ",BK.n_variables_bra)
    print("N variables Ket ",BK.n_variables_ket)
    print("N variables ",BK.n_variables)
    print('Total variables ',BK.total_variables)
    BK.kernel()
    print('opt_res.energy ',BK.opt_res.e)
    print('opt_res.x ',BK.opt_res.x)
    # print('opt_res.x_bra ',BK.opt_res.x_bra)
    # print('opt_res.x_ket ',BK.opt_res.x_ket)
    print('tq opt_res.x',[2*i for i in BK.opt_res.x])
    # print('tq opt_res.x_ket',[2*i for i in BK.opt_res.x_bra])
    # print('tq opt_res.x_ket',[2*i for i in BK.opt_res.x_ket])
    print(BK.opt_res)

def prueba_braket():
    print("=============== SPA TEST ========================")
    geo = "H 0. 0. 0. \n H 0. 0. 1.\n H 0. 0. 2. \n H  0. 0. 3."
    mol = tq.Molecule(geometry=geo, basis_set="sto-3g").use_native_orbitals()
    H = mol.make_hamiltonian()
    U1 = mol.make_ansatz('SPA',edges= [(0,1),(2,3)],optimize=True)
    U2 = mol.make_ansatz('SPA',edges= [(0,3),(1,2)],optimize=True)
    res1 = tq.minimize(tq.ExpectationValue(H=H,U=U1),silent=True)#,ftol= 2.220446049250313e-15, gtol= 2.220446049250313e-14, method="L-BFGS-B")
    res2 = tq.minimize(tq.ExpectationValue(H=H,U=U2),silent=True)#,ftol= 2.220446049250313e-15, gtol= 2.220446049250313e-14, method="L-BFGS-B")
    wfn1 = tq.simulate(U1,variables=res1.variables)
    wfn2 = tq.simulate(U2,variables=res2.variables)
    # ang = deepcopy(res1.angles)
    # ang.update(res2.angles)
    # res12r,res12i = tq.BraKet(ket=U1,bra=U2,operator=H)#tq.simulate(,variables=ang)
    # res21r,res21i = tq.BraKet(ket=U2,bra=U1,operator=H)#tq.simulate(,variables=ang)
    # res12r = tq.simulate(res12r,variables=ang)
    # res12i = tq.simulate(res12i,variables=ang)
    # res21r = tq.simulate(res21r,variables=ang)
    # res21i = tq.simulate(res21i,variables=ang)
    # print('TQ <U1|H|U1>=',res1.energy)#,'<--',ang)
    # print('TQ<U2|H|U2>=',res2.energy)#,'<--',ang)
    # print('<U2|H|U1>=',res12r+res12i)
    # print('<U1|H|U2>=',res21r+res21i)
    # print('------------------------------------')
    # a = tq.Variable('a')
    # b = tq.Variable('b')
    # c,h,g = mol.get_integrals()
    # c = mol.integral_manager.constant_term
    # h = mol.integral_manager.one_body_integrals
    # g = mol.integral_manager.two_body_integrals
    # g = g.reorder('chem')
    # g = g.elems
    bk1 = sun.Braket(molecule=mol,indices=[[(0,2),(1,3)],[(4,6),(5,7)]],init_state=tq.gates.X([0,1,4,5]),backend='tcc',variables=U1.extract_variables())#[a,a]
    bk2 = sun.Braket(molecule=mol, indices=[[(0,6),(1,7)],[(2,4),(3,5)]],init_state=tq.gates.X([0,1,2,3]),backend='tcc',variables=U2.extract_variables())#[i[0]for i in ])#[b,b]
    # bk2 = sun.Braket(int1e=h, int2e=g, n_elec=mol.n_electrons, e_core=c,mo_coeff= mol.integral_manager.orbital_coefficients, indices=[[(0,6),(1,7)],[(2,6),(3,7)]],init_state=tq.gates.X([0,1,2,3]),backend='tcc',variables=U2.extract_variables())#[b,b]
    bk3 = sun.Braket(molecule=mol,bra=[[(0,6),(1,7)],[(2,4),(3,5)]],ket=[[(0,2),(1,3)],[(4,6),(5,7)]],init_state_ket=tq.gates.X([0,1,2,3]),init_state_bra=tq.gates.X([0,1,4,5]),backend='tcc',variables_bra=U2.extract_variables(),variables_ket=U1.extract_variables())
    bk2p = tcc.UCC(mol=from_tequila(mol),run_hf= False, run_mp2= False, run_ccsd= False, run_fci= False,init_method="zeros")
    bk1.minimize()
    bk2.minimize()
    bkwfn1 = tq.simulate(U1,variables=bk1.variables)
    bkwfn2 = tq.simulate(U2,variables=bk2.variables)
    bk2p.ex_ops = bk2.BK.ex_ops_ket
    bk2p.params = bk2.BK.params_ket
    bk2p.init_state = bk2.BK.init_state_ket
    bk2p.kernel()
    print("U2 tccc",bk2p.opt_res.e)
    # print('BK1 Zero ',bk1.simulate([0. for _ in bk1.variables]))
    # print('TQ1 Zero ',tq.simulate(tq.ExpectationValue(H=H,U=U1),variables={d:0. for d in U1.extract_variables()}))
    print('BK2 Zero ',bk2.simulate([0. for _ in bk2.variables]))
    print('TQ2 Zero ',tq.simulate(tq.ExpectationValue(H=H,U=U2),variables={d:0. for d in U2.extract_variables()}))
    print('------------------------------------')
    print('BK <U1|H|U1>=',bk1.opt_res.e)#,' --> ',bk1.variables,'<---',res1.angles.store)#[np.mod(2*i,np.pi) for i in bk1.opt_res.x])
    print('TQ <U1|H|U1>=',res1.energy)
    print('BK U BK VA',bk1.simulate(bk1.opt_res.x))
    print('BK U TQ VA',bk1.simulate(res1.angles))
    print('TQ U BK VA',tq.simulate(tq.ExpectationValue(H=H,U=U1),variables=bk1.variables))
    print('TQ U TQ VA',tq.simulate(tq.ExpectationValue(H=H,U=U1),variables=res1.angles))
    print(f"E1 diff: {(res1.energy-bk1.energy)*1000} mH")
    print('=====================')
    print('BK <U2|H|U2>=',bk2.opt_res.e)#,' --> ',bk2.variables,'<---',res2.angles.store)
    print('TQ <U2|H|U2>=',res2.energy)
    print('BK U BK VA',bk2.simulate(bk2.opt_res.x))
    print('BK U TQ VA',bk2.simulate(res2.angles))
    print('TQ U BK VA',tq.simulate(tq.ExpectationValue(H=H,U=U2),variables=bk2.variables))
    print('TQ U TQ VA',tq.simulate(tq.ExpectationValue(H=H,U=U2),variables=res2.angles))
    print(f"E2 diff: {(res2.energy-bk2.energy)*1000} mH")
    print('=====================')
    for ang in res1.angles.keys():
            if abs(res1.angles[ang]) > 1.e-6 or abs(bk1.variables[ang]) > 1.e-6:
                print(f'{ang} -- Tq {res1.angles[ang]} BK {bk1.variables[ang]} ==> {res1.angles[ang]/bk1.variables[ang]}')
    print('=====================')
    for ang in res2.angles.keys():
            if abs(res2.angles[ang]) > 1.e-6 or abs(bk2.variables[ang]) > 1.e-6:
                print(f'{ang} -- Tq {res2.angles[ang]} BK {bk2.variables[ang]%np.pi} ==> {res2.angles[ang]/bk2.variables[ang]}')
    # print(f"E2 diff: {(res2.energy-bk2p.opt_res.e)*1000} mH")
    # print(bk2p.opt_res)
    # exit()
    # print(f"U1 Ov: {wfn1.inner(bkwfn1)}")
    # print(f"U2 Ov: {wfn2.inner(bkwfn2)}")
    # print("TQ1 ",wfn1)
    # print("BK1 ",bkwfn1)
    # print("TQ2 ",wfn2)
    # print("BK2 ",bkwfn2) #TODO: Parece que BK mantiene todos los signos igual, a lo mejor me estoy dejando alguna convencion
    # # v =res2.angles
    v =bk2.variables
    v.update(bk1.variables)

    # print('<U2|H|U1>=',bk3.simulate(v))
def prueba_uccd():
    print("=============== UCCSD TEST ========================")
    geo = "H 0. 0. 0. \n H 0. 0. 1.\nH 0. 0. 2.\n H 0. 0. 3."
    mol = tq.Molecule(geometry=geo, basis_set="sto-3g")
    H = mol.make_hamiltonian()
    Uref = mol.prepare_reference()
    Ucc = mol.make_uccsd_ansatz(include_reference_ansatz=False)
    U = Uref + Ucc
    exp = tq.ExpectationValue(H=H,U=U)
    res = tq.minimize(exp,silent=True)#,ftol= 2.220446049250313e-15, gtol= 2.220446049250313e-14, method="L-BFGS-B")
    variables = deepcopy(U.extract_variables())
    indices = [i.indices for i in Ucc.gates] 
    variables = [v.extract_variables()[0] for v in Ucc.gates]
    bk1 = sun.Braket(molecule=mol,indices=indices,init_state=Uref,backend='tcc',variables=variables)
    bk1.minimize()
    print('TQ <U1|H|U1>=',res.energy)
    print('BK <U1|H|U1>=',bk1.energy)
    print("TQ VARS ",res.angles.store)
    print("BK VARS ",bk1.variables)
    print('------------------------------------')
    wfn = tq.simulate(U,variables=res.angles)
    bkwfn = tq.simulate(U,variables=bk1.variables)
    print('BK Zero ',bk1.simulate([0. for _ in bk1.variables]))
    print('TQ Zero ',tq.simulate(tq.ExpectationValue(H=H,U=U),variables={d:0. for d in U.extract_variables()}))
    print('BK U BK VA',bk1.simulate(bk1.variables))
    print('BK U TQ VA',bk1.simulate(res.angles))
    print('TQ U BK VA',tq.simulate(tq.ExpectationValue(H=H,U=U),variables=bk1.variables))
    print('TQ U TQ VA',tq.simulate(tq.ExpectationValue(H=H,U=U),variables=res.angles))
    print(f"E1 diff: {(res.energy-bk1.energy)*1000} mH")
    print(f"U Ov: {wfn.inner(bkwfn)}")
    # for ang in res.angles.keys():
    #     if abs(res.angles[ang]) > 1.e-6 or abs(bk1.variables[ang]) > 1.e-6:
    #         print(f'{ang} -- Tq {res.angles[ang]} BK {bk1.variables[ang]} ==> {res.angles[ang]/bk1.variables[ang]}')
    
# prueba_h2()
# prueba_h4()
# prueba_H4_braket()
# prueba_HLi_braket()
# prueba_Variable()
prueba_braket()
# prueba_uccd()
# a = np.array([0,1,2,3,4])
# b = np.array([1,2])
# a.extend(b)
# # d  = {"a":1}
# c  = {"b":2,'e':3,"a":4}
# for i in c.keys():
#     c[i] = c[i]/2
# print(c)
# d.update(c)
# print(d)
# print(a)
# print(a[2:])
# print(a[:2])
# b = None
# c = 0 if b is None else len([1,2,3])
# print(c)
# a = '0000100001'
# print(a.count('1'))
# import tensorcircuit as tc
# U = tc.Circuit(nqubits=4)
# # U.X(0)
# # U.X(1)
# print(U.state())
# print([bin(i) for i in range(len(U.state())) if U.state()[i].real>1.e-6])
# a = tq.Variable("a")
# b = tq.Variable("b")
# l = [a,2*a,b,a*b,1]
# l = tq.Objective(l)
# print(l.extract_variables())
# print(tq.simulate(l,variables={"a":1,'b':2}))
# a = tq.Variable('a')
# b = tq.Variable('b')
# c = 2*a
# d = tq.Variable(c)
# # c.map_variables({'a':1})
# print(a.map_variables({"a":1,"b":2}))
# print(tq.simulate(c,variables={"a":1}))
# print(tq.grad())
# print(a.name)
# print(c.name)
# # print(type(a))
# # print(type(c))
# # print(type(d))
# print(a.extract_variables())
# print(c.extract_variables()[0].name)
# print(c.extract_variables()[1].name)
# print(d.extract_variables())
# print(isinstance(a,tq.Variable))
# print(isinstance(a,tq.Objective))
# print(d.extract_variables())
# a = ['a','b','c']
# for i,j in enumerate(a):
#     print(i,'-->',j)
# a = {"a":1,"a2":2}
# b = {"b":1,"b2":2}
# a.update(b)
# print(a)
# a = tq.Variable('a')
# b = tq.Variable('b')
# p = {"a":1,"b":2}
# va = [1*a,a*b]
# # dE = tq.grad(a*b,"b")
# # print(dE)
# # print(tq.simulate(dE,variables=p))

# ten = tq.QTensor(objective_list=va,shape=(len(va)))
# veinte = ten.apply(tq.grad)
# print(tq.simulate(ten,variables=p))
# print(tq.simulate(veinte,variables=p))
# a = [(0,2),(1,3)]
# b = [(1,3),(0,2)]

# print(all([i in b for i in a]))
# print(all([i in a for i in b]))