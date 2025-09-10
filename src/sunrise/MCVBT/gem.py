import numpy as np
import scipy

from sunrise.expval.fqe_expval import FQEBraKet
from tequila.quantumchemistry import QuantumChemistryBase


def gem_fast(circuits, solver, variables, mol: QuantumChemistryBase, silent=True):
    """
    Fast implementation of tq.apps.gem
    works only with qulacs backend
    not differentiable
    """

    SS = np.eye(len(circuits))
    EE = np.eye(len(circuits))

    for i in range(len(circuits)):
        for j in range(i,len(circuits)):

            if solver == "tcc":
                raise NotImplementedError
            elif solver == "openfermion":
                f = FQEBraKet(ket_fcircuit=circuits[i], bra_fcircuit=circuits[j], molecule=mol)

                ff = FQEBraKet(ket_fcircuit=circuits[i], bra_fcircuit=circuits[j], n_ele=mol.n_electrons,
                               n_orbitals=mol.n_orbitals)
            else:
                raise ValueError("Unknown solver {}".format(solver))

            EE[i,j] = f(variables)
            EE[j,i] = EE[i,j]

            SS[i,j] = ff(variables)
            SS[j,i] = SS[i,j]
    print(EE, "EE")
    print("======================")
    print(SS, "SS")
    print("_____________________")

    v,vv = scipy.linalg.eigh(a=EE,b=SS)


    return v,vv