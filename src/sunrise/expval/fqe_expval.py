from typing import Union, Tuple

import numpy as np
import tequila as tq
from tequila import QubitWaveFunction
from tequila.quantumchemistry import QuantumChemistryBase
import fqe
from tequila.objective import Objective


from sunrise.expval.fqe_utils import *
from sunrise.fermionic_excitation.circuit import FCircuit

from typing import List, Any


class FQEBraKet:

    def __init__(self,
                 ket_fcircuit: FCircuit = None, bra_fcircuit: FCircuit = None,
                 one_body_integrals: Any = None, two_body_integrals = None, constant: int = None,
                 mol: QuantumChemistryBase= None,
                 *args, **kwargs
                 ):

        if "ket_fcircuit" in kwargs and ket_fcircuit is None:
            ket_fcircuit = kwargs["ket_fcircuit"]
            kwargs.pop("ket_fcircuit")
        elif "ket_fcircuit" in kwargs and ket_fcircuit is not None:
            raise ValueError("Two circuits provided")
        if ket_fcircuit is None:
            raise ValueError("No ket fcircuit provided")

        if "bra_fcircuit" in kwargs and bra_fcircuit is None:
            bra_fcircuit = kwargs["bra_fcircuit"]
            kwargs.pop("bra_fcircuit")

        molecule_flag = False
        if ("molecule" in kwargs) and (mol is None):
            mol = kwargs["molecule"]
            kwargs.pop("molecule")
            molecule_flag = True
        elif "molecule" in kwargs and mol is not None:
            raise ValueError("Two Molecules provided")

        if "one_body_integrals" in kwargs and one_body_integrals is None:
            one_body_integrals = kwargs["one_body_integrals"]
            kwargs.pop("one_body_integrals")
        elif "one_body_integrals" in kwargs and one_body_integrals is not None:
            raise ValueError("Two one body integrals provided")

        if "two_body_integrals" in kwargs and two_body_integrals is None:
            two_body_integrals = kwargs["two_body_integrals"]
            kwargs.pop("two_body_integrals")
        elif "two_body_integrals" in kwargs and two_body_integrals is not None:
            raise ValueError("Two two body integrals provided")

        if "constant" in kwargs and constant is None:
            constant = kwargs["constant"]
            kwargs.pop("constant")
        elif "constant" in kwargs and constant is not None:
            raise ValueError("Two constants provided")



        if (one_body_integrals is None and two_body_integrals is not None) \
                or one_body_integrals is not None and two_body_integrals is None:
            raise TequilaException("Both integrals are needed two conunstract a Hamiltonian")
        if (one_body_integrals is not None) and (two_body_integrals is not None) and (constant is None):
            raise TequilaException("Constant not defined")

        if one_body_integrals is not None and two_body_integrals is not None and constant is not None:
            integral_flag = True
        else:
            integral_flag = False

        construct_ham= True
        if integral_flag is False and molecule_flag is False:
            construct_ham = False

        if construct_ham:
            if integral_flag is True:
                self.h_of = make_fermionic_hamiltonian(one_body_integrals, two_body_integrals, constant,)
                self.h_fqe = fqe.get_hamiltonian_from_openfermion(self.h_of)
                self.n_orbitals = one_body_integrals.shape[0]
            elif molecule_flag is True:
                c,h,g = mol.get_integrals()
                self.h_of = make_fermionic_hamiltonian(one_body_integrals=h, two_body_integrals=g.elems, constant=c)
                self.h_fqe = fqe.get_hamiltonian_from_openfermion(self.h_of)
                self.n_orbitals = h.shape[0]
                n_ele = mol.n_electrons
        else:
            self.h_fqe = None
            self.n_orbitals = kwargs.get("n_orbitals")
            n_ele = kwargs.get("n_ele")

        if self.n_orbitals is None:
            raise TequilaException("n_orbitals not defined in the kwargs")

        if n_ele > self.n_orbitals:
            raise TequilaException("number of electrons must not be greater than number of orbitals")

        self.n_ele = n_ele
        ket_fcircuit = ket_fcircuit.to_udud(norb=self.n_orbitals)

        self.parameter_map_ket = []
        for x in ket_fcircuit.variables:
            self.parameter_map_ket.append(tq.assign_variable(x))
        self.parameter_map_ket = list(dict.fromkeys(self.parameter_map_ket))

        # self.parameter_map_ket = [tq.assign_variable(x) for x in ket_fcircuit.variables]

        ket_instructions = ket_fcircuit.extract_indices()
        ket_angles = ket_fcircuit.variables
        self.ket_generator = create_fermionic_generators(ket_instructions, ket_angles)

        bin_dict = generate_of_binary_dict(self.n_orbitals, self.n_ele // 2)

        self.ket = fqe.Wavefunction(param=[[self.n_ele, 0, self.n_orbitals]])  # probably only works for H

        if ket_fcircuit.initial_state is None:
            self.ket.set_wfn(strategy='hartree-fock')
        else:
            set_init_state(wfn = self.ket, n_ele=self.n_ele,n_orb=self.n_orbitals, init_state=ket_fcircuit.initial_state, bin_dict=bin_dict)


        bra_instructions = None
        init_bra = None
        if bra_fcircuit is not None:
            bra_fcircuit = bra_fcircuit.to_udud(norb=self.n_orbitals)
            # self.parameter_map_bra = [tq.assign_variable(x[0]) for x in bra_fcircuit.variables]
            self.parameter_map_bra = []
            for x in bra_fcircuit.variables:
                    self.parameter_map_bra.append(tq.assign_variable(x))
            self.parameter_map_bra = list(dict.fromkeys(self.parameter_map_bra))
            bra_instructions = bra_fcircuit.extract_indices()
            bra_angles = bra_fcircuit.variables
            self.bra_generator = create_fermionic_generators(bra_instructions, bra_angles)
            init_bra = bra_fcircuit.initial_state

        self.bra_instructions = bra_instructions
        self.init_bra = init_bra


        if bra_fcircuit is None:
            self.bra = None
        else:
            self.bra = fqe.Wavefunction(param=[[self.n_ele, 0, self.n_orbitals]])

            if self.init_bra is None:
                self.bra.set_wfn(strategy='hartree-fock')
            else:
                set_init_state(wfn=self.bra, n_ele=self.n_ele,n_orb=self.n_orbitals, init_state=init_bra, bin_dict=bin_dict)

        self.ket_time_evolved = None
        self.bra_time_evolved = None


    def __call__(self,variables, *args, **kwargs):

        if self.bra is not None:
            parameter_map = self.parameter_map_bra + self.parameter_map_ket
        else:
            parameter_map = self.parameter_map_ket
        if type(variables) is not dict:
            variables = {parameter_map[i]: variables[i] for i in range(len(variables))}

        variables = tq.format_variable_dictionary(variables)

        parameters_ket = [x(variables) for x in self.parameter_map_ket]

        zip_ket = zip(parameters_ket, self.ket_generator.values())
        ket_t = self.ket
        for argument in zip_ket:
            ket_t = ket_t.time_evolve(0.5 * argument[0], argument[1])
        #bra time evolution
        if self.bra_instructions is  None:
            bra_t = None
        else:
            parameters_bra = [x(variables) for x in self.parameter_map_bra]
            zip_bra = zip(parameters_bra, self.bra_generator.values())
            bra_t = self.bra
            for argument in zip_bra:
                bra_t = bra_t.time_evolve(0.5 * argument[0], argument[1])

        self.ket_time_evolved = ket_t
        self.bra_time_evolved = bra_t

        if self.h_fqe is not None:
            result = fqe.expectationValue(wfn=ket_t, ops=self.h_fqe, brawfn=bra_t)
        else:
            result = fqe.dot(bra_t, ket_t)

        return result.real


    def print_ket(self):
        self.ket.print_wfn()

    def print_bra(self):
        self.bra.print_wfn()

    def print_ket_time_evolved(self):
        self.ket_time_evolved.print_wfn()

    def print_bra_time_evolved(self):
        self.bra_time_evolved.print_wfn()

    def print_generator(self):
        print(self.ket_generator)


def set_init_state( wfn: fqe.Wavefunction, n_ele, n_orb,
                    init_state: Union[List[Union[Tuple[str, int], QubitWaveFunction, np.array]],QubitWaveFunction],
                    bin_dict: dict) -> None:

    coeff = wfn.get_coeff((n_ele, 0))

    if isinstance(init_state, Tuple):
        for state in init_state:
            if len(state[0]) != n_ele:
                raise TequilaException("initial state is to long")
            n_ones = 0
            for binary in state[0]:
                if binary == "1":
                    n_ones += 1
            if n_ones != n_ele // 2:
                raise TequilaException("initial state has to many ones for the reordrerd JW")

        for state in init_state:
            i = bin_dict[state[0]]
            coeff[i][i] += state[1]
        wfn.set_wfn(strategy="from_data", raw_data={(n_ele, 0): coeff})
        wfn.normalize()

    elif isinstance(init_state, QubitWaveFunction):
        indices, values = init_state_from_wavefunction(wvf=init_state,n_orb=n_orb, bin_dict=bin_dict)
        for i,index in enumerate(indices):
            coeff[index][index] = values[i]

        wfn.set_wfn(strategy="from_data", raw_data={(n_ele, 0): coeff})
        wfn.normalize()

    elif isinstance(init_state, np.ndarray):
        wfn.set_wfn(strategy="from_data", raw_data={(n_ele, 0): init_state[0]})
        wfn.normalize()

    else:
        raise TequilaException("unkown intitial state type {}".format(type(init_state[0])))



def init_state_from_wavefunction(wvf:QubitWaveFunction, n_orb:int, bin_dict:dict):

    indices=[]
    values=[]
    for idx, i in enumerate(wvf._state): #todo check if reversed is always needed
        if abs(i) > 1e-3: #todo check num of ones somewhere
            vec = (bin(idx)[2:])

            if len(vec) < n_orb:
                vec = '0'*(n_orb-len(vec))+vec
            if len(vec) > n_orb:
                vec = vec[:n_orb]
            vec = vec[::-1]
            # vec = vec[len(vec)//2:]
            indices.append(bin_dict[vec])
            values.append(abs(i)) #todo not sure about this


    return indices, values





