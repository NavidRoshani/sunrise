# import tequila as tq
# import sunrise as sun
# import numpy as np
# from numbers import Number
# from scipy.optimize import minimize
# # import tencirchem as tcc
# # from tencirchem.utils.optimizer import soap
# from tequila.utils.bitstrings import BitString, reverse_int_bits
# # from sunrise.expval.tcc_engine.braket import EXPVAL
# from sunrise.expval.pyscf_molecule import from_tequila
# from copy import deepcopy
# import matplotlib.pyplot as plt
# from sunrise.expval import Braket#,from_Qcircuit
# from numpy import isclose

# def from_edges(edges,nmo):
#     '''
#     Expected edges like [(0,1),(2,3),...]
#     Returned [(0,0+nno,...,1+nmo,2),(a,c,...,d,b)]
#     '''

#     assert isinstance(edges,(list,tuple))
#     if isinstance(edges[0],Number):
#         edges = [edges]
#     indices = []
#     for edge in edges:
#         if len(edge) == 1:
#             print(f"Include the edge {edge} on the reference state.")
#         ext = []
#         for i in range(len(edge)-1):
#             ext.append((2*edge[i],2*edge[i+1]))
#             ext.append((2*edge[i]+1,2*edge[i+1]+1))
#         indices.append(ext)
#     print('Indices ',indices )
#     ex_ops = []
#     params = []
#     param_ids = []
#     for exct in indices:
#         exc = []
#         params.append(str(exct))
#         param_ids.append(len(param_ids))
#         for idx in exct:
#             exc.append(idx[0]//2+(idx[0]%2)*nmo)
#             exc.insert(0,idx[1]//2+(idx[1]%2)*nmo)
#         ex_ops.append(tuple(exc))
#     return ex_ops,params,param_ids
# def from_indices(indices,nmo):
#     '''
#     Expected indices like [[(0,2),(1,3),...],[(a,b),(c,d),...],...]
#     Returned [(0,0+nno,...,1+nmo,2),(a,c,...,d,b)]
#     '''
#     assert isinstance(indices,(list,tuple))
#     if isinstance(indices[0],Number):
#         indices = [indices]
#     ex_ops = []
#     params = []
#     param_ids = []
#     for exct in indices:
#         exc = []
#         params.append(str(exct))
#         param_ids.append(len(param_ids))
#         for idx in exct:
#             exc.append(idx[0]//2+(idx[0]%2)*nmo)
#             exc.insert(0,idx[1]//2+(idx[1]%2)*nmo)
#         ex_ops.append(tuple(exc))
#     return ex_ops,params,param_ids
# def from_wavefunction(wvf):
#     init_state = []
#     for i in wvf._state:
#         # print(bin(i),wvf._state[i])
#         vec = bin(i)[2:]
#         vup = ''
#         vdw = ''
#         for j in range(len(vec)//2):
#             vup += vec[2*j]
#             vdw += vec[2*j+1]
#         vec = (vup + vdw)[::-1]
#         init_state.append([vec,wvf._state[i]])
#     return init_state
# def from_simulated(wvf,tol=1e-6):
#     init_state = []
#     nq = wvf.n_qubits
#     nq = int(2* (np.ceil(nq/2))) #if HCB may be odd amount of qubits
#     for i,idx in enumerate(wvf._state):
#         vec = BitString.from_int(i)
#         vec.nbits = nq
#         vec = vec.binary
#         vup = ''
#         vdw = ''
#         for j in range(len(vec)//2):
#             vup += vec[2*j]
#             vdw += vec[2*j+1]
#         vec = (vup + vdw)
#         if idx > tol:
#             if idx.imag > tol: raise tq.TequilaException(f'TCC only support real wvf coefficients, received: {idx}')
#             init_state.append([vec,idx.real])
#     return init_state

# def prueba_h2():
#     print("__________H2_________")
#     geo = "H 0. 0. 0. \n H 0. 0. 1."
#     mol = tq.Molecule(geometry=geo, basis_set="sto-3g")
#     fci = mol.compute_energy(method="fci")
#     print('FCI ', fci)
#     hf = mol.compute_energy(method="hf")
#     print("HF ", hf)
#     mol = mol.use_native_orbitals()
#     U = mol.make_ansatz('HCB-SPA', edges=[(0, 1)])
#     init_guess = np.eye(mol.n_orbitals)
#     init_guess[0, 0] = 1.
#     init_guess[1, 0] = 1.
#     init_guess[0, 1] = -1.
#     init_guess[1, 1] = 1.
#     opt = tq.quantumchemistry.optimize_orbitals(molecule=mol, circuit=U, initial_guess=init_guess, silent=True,use_hcb=True)
#     print(vars(opt).keys())
#     print(opt.mcscf_local_data.keys())
#     print(opt.energy)
#     # print(opt.re)
#     exit()
#     H = opt.molecule.make_hamiltonian()
#     SPA = tq.minimize(tq.ExpectationValue(H=H, U=U + mol.hcb_to_me()), silent=True)
#     print("SPA E: ", SPA.energy)
#     c, h, g = mol.get_integrals()
#     g = g.reorder('chem')
#     g = g.elems
#     s = opt.molecule.integral_manager.overlap_integrals
#     U = tcc.UCC.from_integral(int1e=h, int2e=g,n_elec= mol.n_electrons, e_core=c,ovlp=s,**{"mo_coeff": opt.mo_coeff, "init_method": "zeros", "run_hf": True, "run_mp2": False, "run_ccsd": False, "run_fci": False})
#     exp_ops, params, param_ids = from_indices([((0, 2), (1, 3))], mol.n_orbitals)
#     U.ex_ops = exp_ops
#     U.params = params
#     U.param_ids = param_ids
#     U.civector_fci = tq.QubitWaveFunction.from_string('|1100>').to_array()[::-1]
#     print('ex_ops ', U.ex_ops)
#     print('params', U.params)
#     print('param_ids ', U.param_ids)
#     print(U.get_init_state_dataframe())
#     U.kernel()
#     print('kernel ', U.kernel())
#     print('Ref energy', U.energy([0] * len(param_ids)))

# def prueba_h4():
#     print("__________H4_________")
#     geo = "H 0. 0. 0. \n H 0. 0. 1.\nH 0. 0. 2.\n H 0. 0. 3."
#     mol = tq.Molecule(geometry=geo, basis_set="sto-3g")
#     fci = mol.compute_energy(method="fci")
#     print('FCI ', fci)
#     hf = mol.compute_energy(method="hf")
#     print("HF ", hf)
#     mol = mol.use_native_orbitals()
#     U = mol.make_ansatz('HCB-SPA', edges=[(0, 1), (2, 3)])
#     init_guess = np.eye(mol.n_orbitals)
#     init_guess[0, 0] = 1.
#     init_guess[1, 0] = 1.
#     init_guess[0, 1] = -1.
#     init_guess[1, 1] = 1.
#     init_guess[2, 2] = 1.
#     init_guess[3, 2] = 1.
#     init_guess[2, 3] = -1.
#     init_guess[3, 3] = 1.
#     opt = tq.quantumchemistry.optimize_orbitals(molecule=mol, circuit=U, initial_guess=init_guess, silent=True,
#                                                 use_hcb=True)
#     H = opt.molecule.make_hamiltonian()
#     SPA = tq.minimize(tq.ExpectationValue(H=H, U=U + mol.hcb_to_me()), silent=True)
#     print("SPA E: ", SPA.energy)
#     print('With variables: ',SPA.variables)
#     c, h, g = mol.get_integrals()
#     g = g.reorder('chem')
#     g = g.elems
#     s = opt.molecule.integral_manager.overlap_integrals
#     U = tcc.UCC.from_integral(int1e=h, int2e=g, n_elec=mol.n_electrons, e_core=c, ovlp=s,
#                               **{"mo_coeff": opt.mo_coeff, "init_method": "zeros", "run_hf": True, "run_mp2": False,
#                                  "run_ccsd": False, "run_fci": False})
#     exp_ops, params, param_ids = from_indices([((0, 2), (1, 3)), ((4, 6), (5, 7))], mol.n_orbitals)
#     init_guess = from_wavefunction(tq.QubitWaveFunction.from_string('|11001100>').normalize())
#     # print(init_guess.to_array())
#     for i in init_guess:
#         i[0] = U.get_addr(i[0])
#     init = [0] * U.civector_size
#     for i in init_guess:
#         init[i[0]] = i[1]
#     U.init_state = init
#     # print(U.civector_size)
#     # exit()
#     # print(U.get_init_state_dataframe())
#     U.ex_ops = exp_ops
#     U.params = params
#     U.param_ids = param_ids
#     print('ex_ops ', U.ex_ops)
#     print('params', U.params)
#     print('param_ids ', U.param_ids)
#     print('Init variables ',U.init_guess)
#     print('kernel ', U.kernel())
#     U.kernel()
#     print('Ref energy', U.energy([0] * len(param_ids)))
#     print('Variables ',U.init_guess)
#     # opt_res = minimize(U.energy, U.init_guess, method=soap)
#     # print('Optimized ',opt_res.fun)
#     # print('Opt Vars ',[2*i for i in opt_res.x])

# def prueba_H4_braket():
#     print("__________H4_________")
#     geo = "H 0. 0. 0. \n H 0. 0. 1.\nH 0. 0. 2.\n H 0. 0. 3."
#     mol = tq.Molecule(geometry=geo, basis_set="sto-3g")
#     fci = mol.compute_energy(method="fci")
#     print('FCI ', fci)
#     hf = mol.compute_energy(method="hf")
#     print("HF ", hf)
#     mol = mol.use_native_orbitals()
#     U = mol.make_ansatz('HCB-SPA', edges=[(0, 1), (2, 3)])
#     init_guess = np.eye(mol.n_orbitals)
#     init_guess[0, 0] = 1.
#     init_guess[1, 0] = 1.
#     init_guess[0, 1] = -1.
#     init_guess[1, 1] = 1.
#     init_guess[2, 2] = 1.
#     init_guess[3, 2] = 1.
#     init_guess[2, 3] = -1.
#     init_guess[3, 3] = 1.
#     opt = tq.quantumchemistry.optimize_orbitals(molecule=mol, circuit=U, initial_guess=init_guess, silent=True,
#                                                 use_hcb=True)
#     H = opt.molecule.make_hamiltonian()
#     SPA = tq.minimize(tq.ExpectationValue(H=H, U=U + mol.hcb_to_me()), silent=True)
#     print("SPA E: ", SPA.energy)
#     print('With variables: ',SPA.variables)
#     c, h, g = mol.get_integrals()
#     g = g.reorder('chem')
#     g = g.elems
#     # print('Given ',g)
#     s = opt.molecule.integral_manager.overlap_integrals
#     U = tcc.UCC.from_integral(int1e=h, int2e=g, n_elec=mol.n_electrons, e_core=c, ovlp=s,engine='civector',
#                               **{"mo_coeff": opt.mo_coeff, "init_method": "zeros", "run_hf": True, "run_mp2": False,
#                                  "run_ccsd": False, "run_fci": False})
#     exp_ops, params, param_ids = from_indices([((0, 2), (1, 3)), ((4, 6), (5, 7))], mol.n_orbitals)
#     init_guess = from_wavefunction(tq.QubitWaveFunction.from_string('|11001100>').normalize())
#     # print(init_guess.to_array())
#     for i in init_guess:
#         i[0] = U.get_addr(i[0])
#     init = [0] * U.civector_size
#     for i in init_guess:
#         init[i[0]] = i[1]
#     U.init_state = init
#     # print(U.civector_size)
#     # exit()
#     # print(U.get_init_state_dataframe())
#     U.ex_ops = exp_ops
#     U.params = params
#     U.param_ids = param_ids
#     print('ex_ops ', U.ex_ops)
#     print('params', U.params)
#     print('param_ids ', U.param_ids)
#     print('Init variables ',U.init_guess)
#     print('Ci Vector ',U.get_ci_strings(),'->',len(U.get_ci_strings()))
#     # print('kernel ', U.kernel())
#     U.kernel()
#     print('Ref energy', U.energy([0] * len(param_ids)))
#     psi = U.civector()
#     # U.apply_excitation()
#     bra = U.civector()
#     print(bra.conj() @ psi)
#     print(type(bra.conj() @ psi))
#     # print(mol.make_hamiltonian().to_matrix())
#     # print('->',mol.make_hamiltonian().to_matrix().shape)
#     # U.make_rdm1

# def prueba_HLi_braket():
#     # geo = "H 0. 0. 0. \n H 0. 0. 1.\nH 0. 0. 2.\n H 0. 0. 3."
#     geo = "H 0. 0. 0. \n Li 0. 0. 1."
#     mol = tq.Molecule(geometry=geo, basis_set="sto-3g")
#     H = mol.make_molecular_hamiltonian()
#     c,h,g = mol.get_integrals()
#     # print(h.shape)
#     # c,h,g = mol.integral_manager.get_integrals()
#     # print(h.shape)
#     # exit()
#     hmol = sun.Molecule(geometry=geo, basis_set="sto-3g")
#     fci = mol.compute_energy(method="fci")
#     print('FCI ', fci)
#     hf = mol.compute_energy(method="hf")
#     print("HF ", hf)
#     mol = mol.use_native_orbitals()
#     edges = hmol.get_spa_edges()
#     U = mol.make_ansatz('HCB-SPA', edges=edges)
#     # w = tq.simulate(U,variables={va:0. for va in U.extract_variables()})
#     # print(w)
#     # wfn = tq.QubitWaveFunction.from_array(w._state)
#     # init_state = tq.QubitWaveFunction.from_string('|10000000>').normalize()
#     # print('tq.simulate ',w._state)
#     # print('Qwfn ',init_state._state)
#     # print('Qwfn from array',wfn._state)
#     # print(type(wfn._state))
#     # print(type(w._state))
#     # print(type(init_state._state))
#     # # print(from_wavefunction(init_state))
#     # # print(from_simulated(w))
#     # print(w)
#     # exit()
#     init_guess = hmol.get_spa_guess()
#     opt = tq.chemistry.optimize_orbitals(molecule=mol, circuit=U, initial_guess=init_guess.T, silent=True,use_hcb=True)
#     print('Opt ',opt.energy)
#     bra = mol.make_ansatz('SPA', edges=edges)
#     # ket = mol.make_ansatz('SPA', edges=[(0, 3), (1, 2)])
#     H = opt.molecule.make_hamiltonian()
#     print('Tq Hamiltonian: ',H.to_matrix().shape)
#     braket = tq.ExpectationValue(U=bra,H=H)
#     # braket = tq.BraKet(bra=bra,ket=bra,operator=H)
#     # res = tq.minimize(braket[0],silent=True)
#     res = tq.minimize(braket,silent=True)
#     print(tq.simulate(bra,variables=res.variables))    
#     # print(tq.simulate(ket,variables=res.variables))    
#     print("Energy ",res.energy)
#     print("Variables ",res.variables)
#     c, h, g = mol.get_integrals()
#     g = g.reorder('chem')
#     g = g.elems
#     s = opt.molecule.integral_manager.overlap_integrals
#     # pyscfmol = tq.chemistry.QuantumChemistryPySCF.from_tequila(molecule=opt.molecule).pyscf_molecule
#     # U = tcc.UCC(mol=pyscfmol,* {"mo_coeff": opt.mo_coeff, "init_method": "zeros", "run_hf": True, "run_mp2": False,"run_ccsd": False, "run_fci": False})
#     U = tcc.UCC.from_integral(int1e=h, int2e=g, n_elec=mol.n_electrons, e_core=c,**{"mo_coeff": opt.mo_coeff, "init_method": "zeros", "run_hf": True, "run_mp2": False,"run_ccsd": False, "run_fci": False})
#     ex_ops, params, param_ids = from_edges(edges=edges, nmo=mol.n_orbitals)
#     # init_state = from_wavefunction(tq.QubitWaveFunction.from_string('|11001100>').normalize())
#     # print({va:0. for va in bra.extract_variables()})
#     init_state = from_simulated(tq.simulate(bra,variables={va:0. for va in bra.extract_variables()}))
#     print('F ',init_state)
#     # print(init_guess.to_array())
#     for i in init_state:
#         i[0] = U.get_addr(i[0])
#     init = [0] * U.civector_size
#     for i in init_state:
#         init[i[0]] = i[1]
#     U.init_state = init
#     U.ex_ops = ex_ops
#     U.params = params
#     print('init ',init)
#     print('ex_ops ',ex_ops)
#     print('params ',params)
#     print('param_ids ',param_ids)
#     U.kernel()
#     print(U.opt_res.e)
#     print(U.opt_res.x)
#     print([2*i for i in U.opt_res.x])
#     # print(len(U.aslst))
#     print('--------------')
#     BK = EXPVAL.from_integral(int1e=h, int2e=g, n_elec=mol.n_electrons, e_core=c,**{"mo_coeff": opt.mo_coeff, "init_method": "zeros", "run_hf": True, "run_mp2": False,"run_ccsd": False, "run_fci": False})
#     BK.init_state_bra = init
#     BK.init_state = init
#     BK.ex_ops_bra = ex_ops
#     BK.ex_ops = ex_ops
#     BK.params_bra = params
#     BK.params = params
#     BK.param_ids_bra = param_ids
#     BK.param_ids = param_ids
#     BK.kernel()
#     print(BK.opt_res.e)
#     print(BK.opt_res.x)
#     print([2*i for i in BK.opt_res.x])

# def prueba_Variable():
#     geo = "H 0. 0. 0. \n H 0. 0. 1.\nH 0. 0. 2.\n H 0. 0. 3."
#     # geo = "H 0. 0. 0. \n Li 0. 0. 1."
#     mol = tq.Molecule(geometry=geo, basis_set="sto-3g")
#     H = mol.make_molecular_hamiltonian()
#     c,h,g = mol.get_integrals()
#     hmol = sun.Molecule(geometry=geo, basis_set="sto-3g")
#     fci = mol.compute_energy(method="fci")
#     print('FCI ', fci)
#     hf = mol.compute_energy(method="hf")
#     print("HF ", hf)
#     mol = mol.use_native_orbitals()
#     edges = hmol.get_spa_edges()
#     U = mol.make_ansatz('HCB-SPA', edges=edges)
#     init_guess = hmol.get_spa_guess()
#     opt = tq.chemistry.optimize_orbitals(molecule=mol, circuit=U, initial_guess=init_guess.T, silent=True,use_hcb=True)
#     print('Opt ',opt.energy)
#     bra = mol.make_ansatz('SPA', edges=edges)
#     H = opt.molecule.make_hamiltonian()
#     print('Tq Hamiltonian: ',H.to_matrix().shape)
#     braket = tq.ExpectationValue(U=bra,H=H)
#     res = tq.minimize(braket,silent=True)
#     print(tq.simulate(bra,variables=res.variables))    
#     print("Energy ",res.energy)
#     print("Variables ",res.variables)
#     c, h, g = mol.get_integrals()
#     g = g.reorder('chem')
#     g = g.elems
#     U = tcc.UCC.from_integral(int1e=h, int2e=g, n_elec=mol.n_electrons, e_core=c,**{"mo_coeff": opt.mo_coeff, "init_method": "zeros", "run_hf": True, "run_mp2": False,"run_ccsd": False, "run_fci": False})
#     ex_ops, params, param_ids = from_edges(edges=edges, nmo=mol.n_orbitals)
#     init_state = from_simulated(tq.simulate(bra,variables={va:0. for va in bra.extract_variables()}))
#     print('F ',init_state)
#     for i in init_state:
#         i[0] = U.get_addr(i[0])
#     init = [0] * U.civector_size
#     for i in init_state:
#         init[i[0]] = i[1]
#     # U.init_state = init
#     # U.ex_ops = ex_ops
#     print('init ',init)
#     print('ex_ops ',ex_ops)
#     a = tq.Variable('b')
#     ce = tq.Variable('d')
#     # params = tq.Objective([a,2*a])
#     params= [a,a]
#     param_ids=[0,1]
#     # U.params = params
#     # U.param_ids = param_ids
#     # print('params ',params)
#     # print('param_ids ',param_ids)
#     # U.kernel()
#     # print(U.opt_res.e)
#     # print(U.opt_res.x)
#     # print([2*i for i in U.opt_res.x])
#     # print('--------------')
#     # exit()
#     BK = EXPVAL.from_integral(int1e=h, int2e=g, n_elec=mol.n_electrons, e_core=c,**{"mo_coeff": opt.mo_coeff, "init_method": "zeros", "run_hf": True, "run_mp2": False,"run_ccsd": False, "run_fci": False})
#     BK.init_state_bra = init
#     BK.init_state = init
#     BK.ex_ops_bra = ex_ops
#     BK.ex_ops = ex_ops
#     # BK.params = params
#     BK.params_bra = params
#     BK.params_ket = params
#     # BK.param_ids_bra = param_ids
#     # BK.param_ids = param_ids
#     print('params_bra ',BK.params_bra)
#     print('params_ket ',BK.params_ket)
#     print('init_state_ket ',BK.init_state_ket)
#     print('init_state_brat ',BK.init_state_bra)
#     print('variables_bra ',[i.extract_variables() for i in BK.variables_bra])
#     print('variables_ket ',BK.variables_ket)
#     print('param_to_var_bra ',BK.param_to_var_bra)
#     print('param_to_var_ket ',BK.param_to_var_ket)
#     print("N variables Bra ",BK.n_variables_bra)
#     print("N variables Ket ",BK.n_variables_ket)
#     print("N variables ",BK.n_variables)
#     print('Total variables ',BK.total_variables)
#     BK.kernel()
#     print('opt_res.energy ',BK.opt_res.e)
#     print('opt_res.x ',BK.opt_res.x)
#     # print('opt_res.x_bra ',BK.opt_res.x_bra)
#     # print('opt_res.x_ket ',BK.opt_res.x_ket)
#     print('tq opt_res.x',[2*i for i in BK.opt_res.x])
#     # print('tq opt_res.x_ket',[2*i for i in BK.opt_res.x_bra])
#     # print('tq opt_res.x_ket',[2*i for i in BK.opt_res.x_ket])
#     print(BK.opt_res)

# def prueba_braket():
#     print("=============== SPA TEST ========================")
#     geo = "H 0. 0. 0. \n H 0. 0. 1.\n H 0. 0. 2. \n H  0. 0. 3."
#     mol = tq.Molecule(geometry=geo, basis_set="sto-3g").use_native_orbitals()
#     H = mol.make_hamiltonian()
#     U1 = mol.make_ansatz('SPA',edges= [(0,1),(2,3)],optimize=True)
#     U2 = mol.make_ansatz('SPA',edges= [(0,3),(1,2)],optimize=True)
#     res1 = tq.minimize(tq.ExpectationValue(H=H,U=U1),silent=True)
#     res2 = tq.minimize(tq.ExpectationValue(H=H,U=U2),silent=True)
#     wfn1 = tq.simulate(U1,variables=res1.variables)
#     wfn2 = tq.simulate(U2,variables=res2.variables)
#     ang = deepcopy(res1.angles)
#     ang.update(res2.angles)
#     exp12r,exp12i = tq.BraKet(ket=U1,bra=U2,operator=H)#tq.simulate(,variables=ang)
#     exp21r,exp21i = tq.BraKet(ket=U2,bra=U1,operator=H)#tq.simulate(,variables=ang)
#     res12r = tq.simulate(exp12r,variables=ang)
#     res12i = tq.simulate(exp12i,variables=ang)
#     res21r = tq.simulate(exp21r,variables=ang)
#     res21i = tq.simulate(exp21i,variables=ang)
#     # print('TQ <U1|H|U1>=',res1.energy)#,'<--',ang)
#     # print('<U1|H|U2>=',res21r+res21i)
#     # print('------------------------------------')
#     bk1 = sun.Braket(molecule=mol,indices=[[(0,2),(1,3)],[(4,6),(5,7)]],init_state=tq.gates.X([0,1,4,5]),backend='tcc',variables=U1.extract_variables())#[a,a]
#     bk2 = sun.Braket(molecule=mol, indices=[[(0,6),(1,7)],[(2,4),(3,5)]],init_state=tq.gates.X([0,1,2,3]),backend='tcc',variables=U2.extract_variables())#[i[0]for i in ])#[b,b]
#     bk3 = sun.Braket(molecule=mol,bra=[[(0,6),(1,7)],[(2,4),(3,5)]],ket=[[(0,2),(1,3)],[(4,6),(5,7)]],init_state_ket=tq.gates.X([0,1,4,5]),init_state_bra=tq.gates.X([0,1,2,3]),backend='tcc',variables_bra=U2.extract_variables(),variables_ket=U1.extract_variables())
#     bk1.minimize()
#     bk2.minimize()
#     bkwfn1 = tq.simulate(U1,variables=bk1.variables)
#     bkwfn2 = tq.simulate(U2,variables=bk2.variables)
#     print('BK1 Zero ',bk1.simulate([0. for _ in bk1.variables]))
#     print('TQ1 Zero ',tq.simulate(tq.ExpectationValue(H=H,U=U1),variables={d:0. for d in U1.extract_variables()}))
#     print('BK2 Zero ',bk2.simulate([0. for _ in bk2.variables]))
#     print('TQ2 Zero ',tq.simulate(tq.ExpectationValue(H=H,U=U2),variables={d:0. for d in U2.extract_variables()}))
#     print('------------------------------------')
#     print('BK <U1|H|U1>=',bk1.opt_res.e)
#     print('TQ <U1|H|U1>=',res1.energy)
#     print('BK U BK VA',bk1.simulate(bk1.opt_res.x))
#     print('BK U TQ VA',bk1.simulate(res1.angles))
#     print('TQ U BK VA',tq.simulate(tq.ExpectationValue(H=H,U=U1),variables=bk1.variables))
#     print('TQ U TQ VA',tq.simulate(tq.ExpectationValue(H=H,U=U1),variables=res1.angles))
#     print(f"E1 diff: {(res1.energy-bk1.energy)*1000} mH")
#     print('=====================')
#     print('BK <U2|H|U2>=',bk2.opt_res.e)
#     print('TQ <U2|H|U2>=',res2.energy)
#     print('BK U BK VA',bk2.simulate(bk2.opt_res.x))
#     print('BK U TQ VA',bk2.simulate(res2.angles))
#     print('TQ U BK VA',tq.simulate(tq.ExpectationValue(H=H,U=U2),variables=bk2.variables))
#     print('TQ U TQ VA',tq.simulate(tq.ExpectationValue(H=H,U=U2),variables=res2.angles))
#     print(f"E2 diff: {(res2.energy-bk2.energy)*1000} mH")
#     # print('=====================')
#     # for x in res1.angles.keys():
#     #         if abs(res1.angles[x]) > 1.e-6 or abs(bk1.variables[x]) > 1.e-6:
#     #             print(f'{x} -- Tq {res1.angles[x]} BK {bk1.variables[x]} ==> {res1.angles[x]/bk1.variables[x]}')
#     # print('=====================')
#     # for x in res2.angles.keys():
#     #         if abs(res2.angles[x]) > 1.e-6 or abs(bk2.variables[x]) > 1.e-6:
#     #             print(f'{x} -- Tq {res2.angles[x]} BK {bk2.variables[x]} ==> {res2.angles[x]/bk2.variables[x]}')
#     # print('=====================')
#     print(f"U1 Ov: {wfn1.inner(bkwfn1)}")
#     print(f"U2 Ov: {wfn2.inner(bkwfn2)}")
#     print("TQ1 ",wfn1)
#     print("BK1 ",bkwfn1)
#     print("TQ2 ",wfn2)
#     print("BK2 ",bkwfn2)
#     print('=====================')
#     ang = deepcopy(res1.angles)
#     ang.update(res2.angles)
#     print('TQ Zero <U2|H|U1>=',tq.simulate(exp12r,variables={d:0. for d in ang.keys() })+tq.simulate(exp12i,variables={d:0. for d in ang.keys()}))
#     print('BK Zero <U2|H|U1>=',bk3.simulate([0. for _ in bk3.variables]))
#     v = deepcopy(bk1.variables)
#     v.update(deepcopy(bk2.variables))
#     print('TQ <U2|H|U1>=',res12r+res12i)
#     print('BK <U2|H|U1>=',bk3.simulate(v))
#     print(ang.store)
#     print(v)
#     # tqlist = [] 
#     # bklist = []
#     # for i in np.arange(0,6.4,0.2):
#     #     v[[*v.keys()][1]] = i
#     #     ang[[*ang.keys()][-1]] = i
#     #     tqlist.append(tq.simulate(exp12r,variables=ang)+tq.simulate(exp12i,variables=ang))
#     #     bklist.append(bk3.simulate(v))
#     # fig, ax = plt.subplots()
#     # ratio = [tqlist[i]/bklist[i] for i in range(len(tqlist))]
#     # ax.plot(np.arange(0,6.4,0.2), tqlist, linewidth=2, label='tq')
#     # ax.plot(np.arange(0,6.4,0.2), bklist, linewidth=2, label='bk')
#     # # ax.plot(np.arange(0,6.4,0.2), ratio, linewidth=2, label='rat')
#     # ax.legend(fontsize=14)
#     # plt.show()

# def prueba_uccd():
#     print("=============== UCCSD TEST ========================")
#     geo = "H 0. 0. 0. \n H 0. 0. 1.\nH 0. 0. 2.\n H 0. 0. 3."
#     mol = tq.Molecule(geometry=geo, basis_set="sto-3g")
#     H = mol.make_hamiltonian()
#     Uref = mol.prepare_reference()
#     Ucc = mol.make_uccsd_ansatz(include_reference_ansatz=False)
#     U = Uref + Ucc
#     exp = tq.ExpectationValue(H=H,U=U)
#     res = tq.minimize(exp,silent=True)#,ftol= 2.220446049250313e-15, gtol= 2.220446049250313e-14, method="L-BFGS-B")
#     variables = deepcopy(U.extract_variables())
#     indices = [i.indices for i in Ucc.gates] 
#     variables = [v.extract_variables()[0] for v in Ucc.gates]
#     bk1 = sun.Braket(molecule=mol,indices=indices,init_state=Uref,backend='tcc',variables=variables)
#     bk1.minimize()
#     print('TQ <U1|H|U1>=',res.energy)
#     print('BK <U1|H|U1>=',bk1.energy)
#     print("TQ VARS ",res.angles.store)
#     print("BK VARS ",bk1.variables)
#     print('------------------------------------')
#     wfn = tq.simulate(U,variables=res.angles)
#     bkwfn = tq.simulate(U,variables=bk1.variables)
#     print('BK Zero ',bk1.simulate([0. for _ in bk1.variables]))
#     print('TQ Zero ',tq.simulate(tq.ExpectationValue(H=H,U=U),variables={d:0. for d in U.extract_variables()}))
#     print('BK U BK VA',bk1.simulate(bk1.variables))
#     print('BK U TQ VA',bk1.simulate(res.angles))
#     print('TQ U BK VA',tq.simulate(tq.ExpectationValue(H=H,U=U),variables=bk1.variables))
#     print('TQ U TQ VA',tq.simulate(tq.ExpectationValue(H=H,U=U),variables=res.angles))
#     print(f"E1 diff: {(res.energy-bk1.energy)*1000} mH")
#     print(f"U Ov: {wfn.inner(bkwfn)}")
#     # for ang in res.angles.keys():
#     #     if abs(res.angles[ang]) > 1.e-6 or abs(bk1.variables[ang]) > 1.e-6:
#     #         print(f'{ang} -- Tq {res.angles[ang]} BK {bk1.variables[ang]} ==> {res.angles[ang]/bk1.variables[ang]}')

# def prueba_zero():
#     print("=============== SPA TEST ========================")
#     geo = "H 0. 0. 0. \n H 0. 0. 1."#\n H 0. 0. 2. \n H  0. 0. 3."
#     mol = tq.Molecule(geometry=geo, basis_set="sto-3g").use_native_orbitals()
#     H = mol.make_hamiltonian()
#     # U1 = mol.make_ansatz('SPA',edges= [(0,1),(2,3)],optimize=True)
#     U1 = tq.gates.X([0,1])
#     # U2 = mol.make_ansatz('SPA',edges= [(0,3),(1,2)],optimize=True)
#     U2 = tq.gates.X([1,2])
#     exp12r,exp12i = tq.BraKet(ket=U1,bra=U2,operator=H)
#     exp21r,exp21i = tq.BraKet(ket=U2,bra=U1,operator=H)
#     zeros = {d:0. for d in U1.extract_variables()}
#     zeros.update({d:0. for d in U2.extract_variables()})
#     res12r = tq.simulate(exp12r,variables=zeros)
#     res12i = tq.simulate(exp12i,variables=zeros)
#     res21r = tq.simulate(exp21r,variables=zeros)
#     res21i = tq.simulate(exp21i,variables=zeros)
#     bk1 = sun.Braket(molecule=mol,indices=[[(0,2)]],init_state=U1,backend='tcc',variables=[0.])
#     bk2 = sun.Braket(molecule=mol, indices=[[(0,2)]],init_state=U2,backend='tcc',variables=[0.])
#     bk3 = sun.Braket(molecule=mol,bra=[[(0,2)]],ket=[[(0,2)]],init_state_ket=U1,init_state_bra=U2,backend='tcc',variables_bra=[0.],variables_ket=[0.])
#     print('BK1 <U1|H|U1> Zero ',bk1.simulate([0. for _ in bk1.variables]))
#     print('TQ1 <U1|H|U1> Zero ',tq.simulate(tq.ExpectationValue(H=H,U=U1),variables={d:0. for d in U1.extract_variables()}))
#     print('BK2 <U2|H|U2> Zero ',bk2.simulate([0. for _ in bk2.variables]))
#     print('TQ2 <U2|H|U2> Zero ',tq.simulate(tq.ExpectationValue(H=H,U=U2),variables={d:0. for d in U2.extract_variables()}))
#     print('TQ Zero <U2|H|U1>=',res12r+res21i)
#     print('BK Zero <U2|H|U1>=',bk3.simulate([0. for _ in bk3.variables]))

# def prueba_spa():
#     # geom = "H 0. 0. 0.\n H 0. 0. 1.2\n H 0. 0. 2.4\nH 0. 0. 3.6"
#     geom = "Be 0. 0. 1.6\n H 0. 0. 3.2\nH 0. 0. 0."
#     backend='tcc'
#     mol = tq.Molecule(geometry=geom,basis_set='sto-3g',transformation='reordered-jordan-wigner')
#     U = mol.make_ansatz("UpCCSD",hcb_optimization=False)
#     (idx,ref,variables) = from_Qcircuit(U)
#     # print("ref ",ref)
#     print(len(idx),'idx ',idx)
#     print(len(variables),"Variables ",variables)
#     expval = tq.ExpectationValue(H=mol.make_hamiltonian(),U=U)
#     sunval = Braket(molecule=mol,indices=idx,reference=tq.simulate(ref,{}),backend=backend,variables=variables)
#     tqE = tq.minimize(expval,silent=True)
#     sunval.minimize()
#     # print('Init State\n',sunval.BK.get_init_state_dataframe())
#     # print('Post State\n',sunval.BK.get_excitation_dataframe())
#     # print("Zero E Tq  ",tq.simulate(expval,variables={d:0. for d in U.extract_variables()}))
#     # print("Zero E Sun ",sunval.simulate({d:0. for d in U.extract_variables()}))
#     print("Zero E diff ",abs(tq.simulate(expval,variables={d:0. for d in U.extract_variables()})-sunval.simulate({d:0. for d in U.extract_variables()}))*1000)
#     # print('KET',sunval.ket)
#     # print('KET',sunval.params_ket)
#     # print('KET',sunval.init_state_ket)
#     # print('KET',sunval.variables_ket)
#     # print(tqE.angles.store)
#     # print(sunval.variables)
#     for d in tqE.angles.keys():
#         print(f' -{d} -> tq {tqE.angles[d]} -- tcc {sunval.variables[d]}')
#     sunE = sunval.energy
#     tqwfn = tq.simulate(U,tqE.angles)
#     sunwfn = tq.simulate(U,sunval.variables)
#     print("tqE    ",tqE.energy)
#     print("sunE   ",sunE)
#     print("tq sun ",tq.simulate(expval,variables=sunval.variables))
#     print("sun tq ",sunval.simulate(tqE.angles.store))
#     print('Ediff ',abs(tqE.energy-sunE)*1000)
#     print('tq ',tqwfn)
#     print('Sun ',sunwfn)
#     print('Inner ',abs(tqwfn.inner(sunwfn)))
#     # assert np.isclose(tqE.energy,sunE)
#     # assert np.isclose(abs(tqwfn.inner(sunwfn)),1)

# def prueba_tcc():
#     from tencirchem import UCCSD
#     from tencirchem.molecule import h4

#     uccsd = UCCSD(h4)
#     # evaluate various properties based on custom parameters
#     print(uccsd.params)
#     params = np.zeros(uccsd.n_params)
#     print(uccsd.statevector(params))
#     print(uccsd.energy(params))
#     print(uccsd.energy_and_grad(params))
#     print(uccsd.get_excitation_dataframe())
#     print(uccsd.get_init_state_dataframe())

# def prueba_overlap():
#     geom = 'H 0. 0. 0. \n H 0. 0. 1. \n H 0. 0. 2. \n H 0. 0. 3.'
#     mol = tq.Molecule(geometry=geom,basis_set='sto-3g',transformation='reordered-jordan-wigner').use_native_orbitals()
#     H = mol.make_hamiltonian()
#     U1 = mol.make_ansatz("SPA",edges=[(0,1),(2,3)])
#     U2 = mol.make_ansatz("SPA",edges=[(0,2),(1,3)])
#     idx1,ref1,p1 = from_Qcircuit(mol.make_ansatz("SPA",edges=[(0,1),(2,3)],optimize=False))
#     idx2,ref2,p2 = from_Qcircuit(mol.make_ansatz("SPA",edges=[(0,2),(1,3)],optimize=False))
#     sunval = Braket(molecule=mol,bra=idx1,ket=idx2,init_state_bra=ref1,init_state_ket=ref2,backend='tcc',variables_bra=p1,variables_ket=p2)
#     res1 = tq.minimize(tq.ExpectationValue(H=H,U=U1),silent=True)
#     res2 = tq.minimize(tq.ExpectationValue(H=H,U=U2),silent=True)
#     rov,iov = tq.BraKet(bra=U1,ket=U2,H=H)
#     res1.angles.update(res2.angles)
#     tqov = tq.simulate(rov,variables=res1.angles) + tq.simulate(iov,variables=res1.angles)
#     bkov = sunval.simulate(res1.angles)
#     print(tqov)
#     print(bkov)

# def prueba_He():
#     mol = sun.Molecule(geometry='H 0. 0. 0.\nH 0. 0. 1.',basis_set='sto-3g')
#     print(mol.get_spa_edges())
#     print(mol.get_spa_guess())
#     mol = sun.Molecule(geometry='He 0. 0. 0.\nHe 0. 0. 1.',basis_set='sto-3g')
#     print(mol.get_spa_edges())
#     print(mol.get_spa_guess())

# def graph_N2():
#     geo = 'N 0. 0. 0. \n N 0. 0. 1.07'
#     mol = tq.Molecule(geometry=geo,basis_set='sto-3g',transformation='reordered-jordan-wigner').use_native_orbitals()
#     # U = mol.make_ansatz('UpCCSD')
#     hmol = sun.Molecule(geometry=geo,basis_set='sto-3g').use_native_orbitals()
#     # sun.plot_MO(mol)
#     edges = hmol.get_spa_edges()
#     # print(edges)
#     guess = hmol.get_spa_guess()
#     U = mol.make_spa_ansatz(edges=edges,optimize=False,hcb=False)
#     # print(vars(U.gates[-1]))
#     # print(U.make_parameter_map())
#     exit()
#     # print(guess)
#     # print(guess[3])
#     # print(guess[7])
#     opt = tq.chemistry.optimize_orbitals(molecule=mol,circuit=mol.make_ansatz('HCB-SPA',edges),initial_guess=guess.T,use_hcb=True,silent=True)
#     sun.plot_MO(molecule=opt.molecule,filename='a')

# def aprueba_spa():
#     geom = "H 0.0 0.0 0.0\nH 0.0 0.0 1.6\nH 0.0 0.0 3.2\nH 0.0 0.0 4.8"
#     backend = 'tcc'
#     mol = tq.Molecule(geometry=geom,basis_set='sto-3g',transformation='reordered-jordan-wigner')
#     U = mol.make_ansatz("UpCCSD",hcb_optimization=False) #could be hcb-optimized but the from_Qcircuit doesnt work
#     circuit = sun.FCircuit.from_Qcircuit(U)
#     expval = tq.ExpectationValue(H=mol.make_hamiltonian(),U=U)
#     sunval = Braket(molecule=mol,circuit=circuit,backend=backend)
#     # print(sunval.variables)
#     tqE = tq.minimize(expval,silent=True)
#     sunval.minimize()
#     sunE = sunval.energy
#     tqwfn = tq.simulate(U,tqE.angles)
#     sunwfn = tq.simulate(U,sunval.variables)
#     print(tqE.energy,sunE)
#     print(abs(tqwfn.inner(sunwfn)))
#     print(sunval.variables)
#     print(tqE.angles)
#     assert isclose(tqE.energy,sunE)
#     assert isclose(abs(tqwfn.inner(sunwfn)),1,1.e-3)
# def prueba_fixed_variables():
#     geom = "H 0.0 0.0 0.0\nH 0.0 0.0 1.6\nH 0.0 0.0 3.2\nH 0.0 0.0 4.8"
#     backend = 'tcc'
#     mol = tq.Molecule(geometry=geom,basis_set='sto-3g',transformation='reordered-jordan-wigner')
#     U = mol.make_ansatz("SPA",edges=[(0,1),(2,3)],optimize=False)
#     circuit = sun.FCircuit.from_edges([(0,1),(2,3)],n_orb=mol.n_orbitals)
#     U=U.map_variables({d:0.1 for d in U.extract_variables()})
#     circuit=circuit.map_variables({d:0.1 for d in circuit.extract_variables()})
#     expval = tq.ExpectationValue(H=mol.make_hamiltonian(),U=U)
#     sunval = Braket(molecule=mol,circuit=circuit,backend=backend)
#     # print(sunval.variables)
#     tqE = tq.minimize(expval,silent=True)
#     sunval.minimize()
#     sunE = sunval.energy
#     # tqwfn = tq.simulate(U,tqE.angles)
#     # sunwfn = tq.simulate(U,sunval.variables)
#     print('tqE sunE',tqE.energy,sunE)
#     print('tq simulate ',tq.simulate(expval,variables={d:0.1 for d in U.extract_variables()}))
#     print('sun simulate',sunval.simulate({d:0.1 for d in circuit.extract_variables()}))
#     # print(abs(tqwfn.inner(sunwfn)))
#     # print(sunval.variables)
#     # print(tqE.angles)
#     print(sunval())
#     # assert isclose(tqE.energy,sunE)
#     # assert isclose(abs(tqwfn.inner(sunwfn)),1,1.e-3)
# # prueba_h2()
# # prueba_h4()
# # prueba_H4_braket()
# # prueba_HLi_braket()
# # prueba_Variable()
# # prueba_braket()
# # prueba_uccd()
# # prueba_zero()
# # prueba_spa()
# # prueba_tcc()
# # prueba_overlap()
# # prueba_He()
# # graph_N2()
# # aprueba_spa()
# # prueba_fixed_variables()


# # a = 0.1
# # a = tq.assign_variable(a)
# # print(a,type(a))
# # a = 0.2 + a
# # print(a,type(a))
# # print(a.map_variables(),type(a.map_variables()))
# # print(tq.grad(a))
# # print(tq.grad(a))
# # tq.minimize
# # from tequila.circuit.gates
# # from tequila.objective.objective import FixedVariable,Variables
# # v = Variables()
# # v[tq.Variable("a")]=None
# # v[tq.Variable("c")]=None
# # print(v)

# # import sys
# # import pickle

# # mol = tq.Molecule(geometry='H 0.0 0. 0. \n H 0. 0. 1.',basis_set='sto-3g')
# # U = mol.make_ansatz('SPA',edges=[(0,1)])
# # H = mol.make_hamiltonian()
# # res = tq.minimize(tq.ExpectationValue(H=H,U=U))
# # print('E',res.energy,res.angles)
# # with open(sys.path[0]+'/'+'angles.data', 'wb') as file:
# #         pickle.dump(res.angles, file)

# # objects = []
# # with (open(sys.path[0]+'/'+'angles.data', "rb")) as openfile:
# #     while True:
# #         try:
# #             objects.append(pickle.load(openfile))
# #         except EOFError:
# #             break
# # v = objects[0]
# # print('--------')
# # print(v)
# # print(tq.simulate(tq.ExpectationValue(H=H,U=U),variables=v))
# mol = tq.Molecule(geometry='H 0.0 0. 0. \n H 0. 0. 1.',basis_set='sto-3g')
# U = sun.gates.UC(i=0,j=1,variables="a")
# U += sun.gates.UR(i=0,j=1,variables="a")
# U += sun.gates.FermionicExcitation(indices=[(0,4),(1,3)],variables="a")
# U += sun.gates.UX(indices=[(0,4),(1,3)],variables="a")
# U += sun.gates.Phase(i=0,variables="a")
# print(U)
# print(U.gates[0].indices)
# print(U.gates[0].variables)
# print(U.to_qcircuit(molecule=mol))
# a = tq.Variable(0.2)
# a = tq.assign_variable(0.2)
# print(type(a))
# print(2*a)
# print(type(2*a))
# b = tq.Objective([a])
# print(b.extract_variables())
# a = tq.Variable('a')
# b = tq.Variable('c')
# c = tq.Variable('a')

# print(a==b)
# print(a==c)

# U = tq.QCircuit

# import pytest
# from scipy.sparse import linalg
# from pyscf import fci
# from openfermion.linalg import eigenspectrum
# import tensorcircuit as tc

# from tencirchem import UCCSD
# from tencirchem.molecule import h4, h6, _random
# from tencirchem.static.hamiltonian import get_h_from_hf, mpo_to_quoperator
# from tencirchem.static.ci_utils import get_ci_strings
# from tencirchem.static.hea import binary, parity


# def test_hamiltonian(m=h4, mode="fermion",htype= "sparse"):
#     hf = m.HF()
#     hf.chkfile = None
#     hf.verbose = 0
#     hf.kernel()

#     hamiltonian = get_h_from_hf(hf, mode=mode, htype=htype)
#     print(hamiltonian)
#     if htype == "mpo":
#         hamiltonian = mpo_to_quoperator(hamiltonian).eval_matrix()
        
#     else:
#         hamiltonian = np.array(hamiltonian.todense())
#     print(hamiltonian)
#     exit()
#     e_nuc = hf.energy_nuc()
#     if mode in ["fermion", "qubit"]:
#         fci_e, _ = fci.FCI(hf).kernel()
#         # not generally true but works for this example
#         np.testing.assert_allclose(np.linalg.eigh(hamiltonian)[0][0] + e_nuc, fci_e, atol=1e-6)
#     else:
#         circuit = tc.Circuit(4)
#         for i in range(4 // 2):
#             circuit.X(3 - i)
#         state = circuit.state()
#         np.testing.assert_allclose(state @ (hamiltonian @ state).reshape(-1) + e_nuc, hf.e_tot)

# test_hamiltonian()
# import sunrise as sn
# import tequila as tq
# import cProfile
# import numpy
# system = "H 0.0 0.0 0.0\nH 0.0 0.0 1.6\nH 0.0 0.0 3.2\nH 0.0 0.0 4.8"
# select = 'FFFFFFFFFFF'
# excit =  [(0,4)]
# mol_mix = sn.Molecule(geometry=system, basis_set="sto-6g", select=select,condense=False,backend='pyscf',nature='hybrid')
# mol_jw = tq.Molecule(basis_set="sto-6g", geometry=system,backend='pyscf')
# U_jw = mol_jw.prepare_reference()
# U_mix = mol_mix.prepare_reference()

# U_jw += mol_jw.make_excitation_gate(indices=excit, angle="a")
# U_mix += mol_mix.make_excitation_gate(indices=excit, angle="a")

# H_jw = mol_jw.make_hamiltonian()
# H_mix = mol_mix.make_hamiltonian()

# E_jw = tq.ExpectationValue(H=H_jw, U=U_jw)
# E_mix = tq.ExpectationValue(H=H_mix, U=U_mix)

# result_jw = tq.minimize(E_jw,silent=True)
# result_mix = tq.minimize(E_mix,silent=True)

# U_mix += mol_mix.transformation.hcb_to_me(bos=True)
# U_mix.n_qubits=U_jw.n_qubits
# wfn_jw = tq.simulate(U_jw,variables=result_jw.variables)
# wfn_mix = tq.simulate(U_mix,variables=result_mix.variables)
# F = abs(wfn_jw.inner(wfn_mix))

# assert numpy.isclose(F, 1.0, 10 ** -4)
# assert numpy.isclose(result_mix.energy, result_jw.energy, 10 ** -4)






import sunrise as sn
from sunrise.molecules import HyMolecule
import tequila as tq
tq.BraKet
tq.make_transition
tq.make_overlap
def recursion():
    backend = 'tcc'
    geom = "H 0.0 0.0 0.0\nH 0.0 0.0 1.6\nH 0.0 0.0 3.2\nH 0.0 0.0 4.8"
    mol = tq.Molecule(geometry=geom, basis_set='sto-3g',transformation='reordered-jordan-wigner').use_native_orbitals()
    mol = HyMolecule(geometry=geom, basis_set='sto-3g').use_native_orbitals()
    H = mol.make_hamiltonian()
    U1 = mol.make_ansatz("SPA", edges=[(0, 1), (2, 3)])
    U2 = mol.make_ansatz("SPA", edges=[(0, 2), (1, 3)])
    bra = sn.FCircuit.from_edges([(0, 1), (2, 3)], n_orb=mol.n_orbitals)
    ket = sn.FCircuit.from_edges([(0, 2), (1, 3)], n_orb=mol.n_orbitals)
    # ov ,_ = tq.BraKet(bra=U1, ket=U2, H=H)
    # tqS = tq.minimize(ov, silent=True)
    snov = sn.Braket(molecule=mol, ket=ket, bra=bra,backend=backend)
    print(snov)
    snS = sn.minimize(snov,silent=True)
    print(snS.energy)
    print(snS.variables)
    print(snS.angles)
    print(sn.simulate(ket,variables=snS.variables))
    # assert isclose(tqS.energy, snS.energy, atol=1.e-3)
# cProfile.run('recursion()','out.txt')
# recursion()


# geom = "H 0.0 0.0 0.0\nH 0.0 0.0 160"#\nH 0.0 0.0 3.2\nH 0.0 0.0 4.8"
# mol = tq.Molecule(geometry=geom, basis_set='sto-3g')
# print(mol.compute_energy('fci'))
