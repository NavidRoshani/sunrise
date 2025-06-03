import tequila as tq
import numpy as np
import matplotlib.pyplot as plt
from exctsolver import ExcitationSolveScipy
import logging

# logging.basicConfig(level=logging.DEBUG)

mol = tq.Molecule(geometry="H 0 0 0\nLi 0 0 1", frozen_core=False, basis_set="sto-3g")#.use_native_orbitals()
fci_energy = mol.compute_energy(method="fci")
H = mol.make_hamiltonian()
U = mol.make_upccgsd_ansatz(spin_adapt_singles=False,hcb_optimization=False)
print(U)
# U = mol.prepare_reference() + mol.make_excitation_gate(indices=[(0,2),(1,3)],angle="b")+mol.make_excitation_gate(indices=[(0,2)],angle='a')+mol.make_excitation_gate(indices=[(1,3)],angle='c')
# print(U)
# exit()


E = tq.ExpectationValue(H=H, U=U)
E = tq.compile(E)#,backend='scipy')
print(E.extract_variables())
# print(E({v:0 for v in E.extract_variables()}))
def e_with_lst(x):
    return E({E.extract_variables()[i]: val for i, val in enumerate(x)})
    
x0 = np.array([0.0 for i in E.extract_variables()]) # IMPORTANT: use float values here, optimizer cannot work with integer values
print(f"{e_with_lst(x0)=}")


opt = ExcitationSolveScipy(maxiter=10, save_parameters=True)
result = opt.minimize(fun=e_with_lst, x0=x0)
print(result)
print(f"{fci_energy=}")
plt.plot(np.abs(opt.energies - fci_energy), "x")
plt.yscale("log")
plt.savefig("excsolve_opt.png", bbox_inches="tight", dpi=256)

print(opt.params)

x_vals_1d = np.linspace(start=0, stop=4*np.pi, num=100)
idx = 0
# x_vals = np.array([[val] for val in x_vals_1d])
#x_vals = np.array([[0, val] for val in x_vals_1d])
#x_vals = np.array([[val, 0] for val in x_vals_1d])
x_vals = np.array([[val,] + (len(E.extract_variables()) - 1) * [0.5,] for val in x_vals_1d])
print(f"{x_vals.shape=}")
y_vals = [e_with_lst(vals) for vals in x_vals]
plt.figure()
plt.plot(x_vals_1d, y_vals)
plt.savefig("periodicity.png", bbox_inches="tight", dpi=256)

# result = tq.minimize(objective=E,
#                     method="bfgs",
#                     # initial_values=init,
#                     tol=1.e-3,
#                     method_options={"gtol":1.e-3})

# print(result.energy)