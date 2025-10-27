import sunrise as sn
import tequila as tq

geometry = 'H 0. 0. 0. \n H 0. 0. 1. \n H 0. 0. 2.\n H 0. 0. 3.'
mol = sn.Molecule(geometry=geometry, basis_set='sto-3g',nature='f',backend='pyscf').use_native_orbitals()
ref = mol.compute_energy('fci')
# sn.plot_MO(mol,filename='native')
edges = mol.get_spa_edges() # Since H4, it will be [(0,1),(2,3)]
initial_guess = mol.get_spa_guess().T 
U = sn.FCircuit.from_edges(n_orb=mol.n_orbitals, edges=edges)
opt = sn.optimize_orbitals(circuit=U, molecule=mol,silent=True,backend='tcc',initial_guess=initial_guess)
omol = opt.molecule
# sn.plot_MO(omol,filename='first_graph_orbitals')
UR = omol.UR(0,1) + omol.UR(2,3)
UR1 = omol.UR(1,2) + omol.UR(0,3)
UC = omol.UC(1,2) + omol.UC(0,3)
SPAplus = U + UR + UR1 + UC + UR1.dagger() + UR.dagger()
E_spa = sn.minimize(sn.Braket(backend='tcc',circuit=SPAplus, molecule=omol), silent=True)
print("SPA Energy H4:", (opt.energy-ref)*1000)
print('SPA+ Energy H4:', (E_spa.energy-ref)*1000)
sn.plot_MO(,filename='back_to_native')
sn.plot_MO(,filename='second_graph_orbitals')
