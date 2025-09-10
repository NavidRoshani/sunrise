import tequila as tq
import scipy
import numpy as np
from gem import gem_fast
from tequila.quantumchemistry import QuantumChemistryBase

import sunrise as sn
from sunrise.MCVBT.Big_Exp import BigExpVal




class MCVBT:

    def __init__(self, mol: QuantumChemistryBase, graphs: list, circuits=None, strategy=None, solver="FQE"):

        self.mol = mol
        self.graphs = graphs
        self.strategy = strategy
        self.solver = solver
        self.circuits = circuits

        self.results = {}

        if self.circuits is not None:
            if len(self.circuits) != len(self.graphs):
                raise ValueError("Number of circuits must be equal to number of graphs")




    def calculate_groundstate(self):

        if self.circuits is None:
            self.circuits = []
            for i, edges in enumerate(self.graphs):
                U = sn.FCircuit.from_edges(edges=edges, label="G{}".format(i), n_orb=self.mol.n_orbitals)
                for j, e in enumerate(edges):
                    U += sn.gates.FermionicExcitation(indices=[(2 * e[0], 2 * e[1])],
                                                      variables="(R{}_{})".format(i, j))
                    U += sn.gates.FermionicExcitation(indices=[(2 * e[0] + 1, 2 * e[1] + 1)],
                                                      variables="(R{}_{})".format(i, j))

                self.circuits.append(U)

        variables_preopt, energies = self.preoptimize_variables()
        self.results[(1, 0)]=min(energies)

        N = len(self.circuits)
        for n in range(2, N+1):
            v, _ = gem_fast(circuits=self.circuits[:n], solver=self.solver, variables=variables_preopt, mol=self.mol)
            self.results[(n, 0)] = v[0]

        start = 2
        for m in range(1,N+1):
            if m > start:
                start = m
            for n in range(start, N+1):
                v, vv, variables = GNM(circuits=self.circuits[:n], variables=variables_preopt, solver=self.solver, mol=self.mol, silent=True, M=m, max_iter=10)
                self.results[(m, n)] = v[0]


        return self.results

    def compare_to_fci(self):

        fci = self.mol.compute_energy("fci")
        print("G(N,M) errors:")
        for k, v in self.results.items():
            error = abs(fci - v)
            print("G({},{}): {:+2.5f}".format(k[0], k[1], error))


    def preoptimize_variables(self):
        variables_preopt = {}
        energies = []
        for preopt_circuit in self.circuits:
            if self.solver == "FQE":
                E_preopt = sn.expval.FQEBraKet(ket_fcircuit=preopt_circuit, molecule=self.mol)

                variables = preopt_circuit.variables
                init_vars = {vs: 0 for vs in variables}
                x0 = list(init_vars.values())

                r = scipy.optimize.minimize(fun=E_preopt, x0=x0, jac="2-point", method="bfgs",
                                            options={"finite_diff_rel_step": 1.e-5, "disp": False})
                energies.append(r.fun)

                r_variabales = {vs: r.x[i].real for i, vs in enumerate(init_vars)}
                variables_preopt = {**variables_preopt, **r_variabales}

        return variables_preopt, energies



def GNM(circuits, variables, solver, mol, silent=True, max_iter=10, M=None):

    variables = {**variables}
    print(variables)
    print("+++++++++++++++++++++++++++++++++" )
    N = len(circuits)
    if M is None:
        M = len(circuits)

    for i in range(M, N): #map pre opt variables to current circuit set and overrite old circuits
        U = circuits[i]
        U = U.map_variables(variables)
        circuits[i] = U

    vkeys = []
    for U in circuits:
        vkeys += U.extract_variables()

    variables = {**{k: 0.0 for k in vkeys if k not in variables}, **variables}

    v, vv = gem_fast(circuits=circuits, variables=variables, mol=mol, solver=solver)
    x0 = {k: variables[k] for k in vkeys}


    coeffs = []
    for i in range(len(circuits)):
        c = tq.Variable(("c", i))
        coeffs.append(c)
        x0[c] = vv[i, 0]
        vkeys.append(c)

    energy = 1.0
    def callback(x):
        energy = mcvbt_exp(x)
        if not silent:
            print("current energy: {:+2.4f}".format(energy))

    mcvbt_exp = BigExpVal(circuits=circuits, coefficcents=coeffs, mol=mol, solver=solver)
    print(x0)
    for i in range(max_iter):
        result = scipy.optimize.minimize(mcvbt_exp, x0=list(x0.values()), jac="2-point", method="bfgs",
                                         options={"finite_diff_rel_step": 1.e-5, "disp": True}, callback=callback)

        x0 = {vkeys[i]: result.x[i] for i in range(len(result.x))}
        v, vv = gem_fast(circuits=circuits, variables=x0, mol=mol, solver=solver)

        for i in range(len(coeffs)):
            x0[coeffs[i]] = vv[i, 0]

        if np.isclose(energy, v[0], atol=1.e-4):
            print("not converged")
            print(energy)
            print(v[0])
            energy = v[0]
        else:
            energy = v[0]
            break

    for k in vkeys:
        variables[k] = x0[k]

    return v, vv, variables

