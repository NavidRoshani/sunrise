from typing import Union, Tuple

import numpy as np
import tequila as tq
from tequila import QubitWaveFunction
import fqe
from tequila.objective import Objective


from sunrise.expval.fqe_utils import *
from sunrise.fermionic_excitation.circuit import FCircuit

from typing import List, Any


#
# class FQEBraKet:
#
#     def __init__(self,
#                  n_ele: int,
#                  ket_instructions,
#                  bra_instructions = None,
#                  one_body_integrals = None, two_body_integrals = None, constant: int = None,
#                  init_ket: List[Union[Tuple[str, int], QubitWaveFunction, np.array]] = None,
#                  init_bra: List[Union[Tuple[str, int], QubitWaveFunction, np.array]] = None,
#                  *args, **kwargs
#                  ):
#
#         if (one_body_integrals is None and two_body_integrals is not None) \
#                 or one_body_integrals is not None and two_body_integrals is None:
#             raise TequilaException("Both integrals are needed two conunstract a Hamiltonian")
#         if (one_body_integrals is not None) and (two_body_integrals is not None) and (constant is None):
#             raise TequilaException("Constant not defined")
#         construct_ham= True
#
#         if one_body_integrals is None and  two_body_integrals is None:
#             if constant is not None:
#                 print("Constant offset will be ignored")
#             construct_ham = False
#
#         if construct_ham:
#             self.h_of = make_fermionic_hamiltonian(one_body_integrals, two_body_integrals, constant,)
#             self.h_fqe = fqe.get_hamiltonian_from_openfermion(self.h_of)
#             self.n_orbitals = one_body_integrals.shape[0]
#         else:
#             self.h_of = None
#             self.n_orbitals = kwargs.get("n_orbitals")
#         if self.n_orbitals is None:
#             raise TequilaException("n_orbitals not defined in the kwargs")
#
#         if n_ele > self.n_orbitals:
#             raise TequilaException("number of electrons must not be greater than number of orbitlas")
#
#         self.n_ele = n_ele
#
#         self.parameter_map_ket=[tq.assign_variable(x[0]) for x in ket_instructions]
#         ket_instructions = [x[1:] for x in ket_instructions]
#         self.ket_generator = create_fermionic_generators(ket_instructions)
#
#         bin_dict = generate_of_binary_dict(self.n_orbitals, self.n_ele // 2)
#
#         self.ket = fqe.Wavefunction(param=[[self.n_ele, 0, self.n_orbitals]])  # probably only works for H
#         if init_ket is None:
#             self.ket.set_wfn(strategy='hartree-fock')
#         else:
#
#             set_init_state(wfn = self.ket, n_ele=self.n_ele, init_state=init_ket, bin_dict=bin_dict)
#             # i = self.bin_dict[self.init_ket]
#             # ket_coeff = self.ket.get_coeff((self.n_ele, 0))
#             # ket_coeff[i][i] = 1
#             # self.ket.set_wfn(strategy="from_data", raw_data={(self.n_ele, 0): ket_coeff})
#             # self.ket.normalize()
#
#         if bra_instructions is not None:
#             self.parameter_map_bra = [tq.assign_variable(x[0]) for x in bra_instructions]
#             bra_instructions = [x[1:] for x in bra_instructions]
#             self.bra_generator = create_fermionic_generators(bra_instructions)
#
#         self.bra_instructions = bra_instructions
#         self.init_bra = init_bra
#
#
#         if bra_instructions is None:
#             self.bra = None
#         else:
#             self.bra = fqe.Wavefunction(param=[[self.n_ele, 0, self.n_orbitals]])
#
#             if self.init_bra is None:
#                 self.bra.set_wfn(strategy='hartree-fock')
#             else:
#                 set_init_state(wfn=self.bra, n_ele=self.n_ele, init_state=init_bra, bin_dict=bin_dict)
#                 # i = self.bin_dict[self.init_bra]
#                 # bra_coeff = self.bra.get_coeff((self.n_ele, 0))
#                 # bra_coeff[i][i] = 1
#                 # self.bra.set_wfn(strategy="from_data", raw_data={(self.n_ele, 0): bra_coeff})
#
#
#         self.ket_time_evolved = None
#         self.bra_time_evolved = None
#
#
#     def __call__(self,variables, *args, **kwargs):
#
#
#         variables = tq.format_variable_dictionary(variables)
#
#         parameters_ket = [x(variables) for x in self.parameter_map_ket]
#         zip_ket = zip(parameters_ket, self.ket_generator.values())
#         ket_t = self.ket
#         for argument in zip_ket:
#            ket_t = ket_t.time_evolve(0.5 * argument[0], argument[1])
#
#         #bra time evolution
#         if self.bra_instructions is  None:
#             bra_t = None
#         else:
#
#             parameters_bra = [x(variables) for x in self.parameter_map_bra]
#             zip_bra = zip(parameters_bra, self.bra_generator.values())
#             bra_t = self.bra
#             for argument in zip_bra:
#                 bra_t = bra_t.time_evolve(0.5 * argument[0], argument[1])
#
#         self.ket_time_evolved = ket_t
#         self.bra_time_evolved = bra_t
#         result = fqe.expectationValue(wfn=ket_t, ops=self.h_fqe, brawfn=bra_t)
#
#         return result
#
#
#     def print_ket(self):
#         self.ket.print_wfn()
#
#     def print_bra(self):
#         self.bra.print_wfn()
#
#     def print_ket_time_evolved(self):
#         self.ket_time_evolved.print_wfn()
#
#     def print_bra_time_evolved(self):
#         self.bra_time_evolved.print_wfn()
#


class FQEBraKet:

    def __init__(self,
                 n_ele: int,
                 ket_fcircuit: FCircuit, bra_fcircuit: FCircuit = None,
                 one_body_integrals: Any = None, two_body_integrals = None, constant: int = None,
                 *args, **kwargs
                 ):

        if (one_body_integrals is None and two_body_integrals is not None) \
                or one_body_integrals is not None and two_body_integrals is None:
            raise TequilaException("Both integrals are needed two conunstract a Hamiltonian")
        if (one_body_integrals is not None) and (two_body_integrals is not None) and (constant is None):
            raise TequilaException("Constant not defined")
        construct_ham= True

        if one_body_integrals is None and  two_body_integrals is None:
            if constant is not None:
                print("Constant offset will be ignored")
            construct_ham = False

        if construct_ham:
            self.h_of = make_fermionic_hamiltonian(one_body_integrals, two_body_integrals, constant,)
            self.h_fqe = fqe.get_hamiltonian_from_openfermion(self.h_of)
            self.n_orbitals = one_body_integrals.shape[0]
        else:
            self.h_of = None
            self.n_orbitals = kwargs.get("n_orbitals")
        if self.n_orbitals is None:
            raise TequilaException("n_orbitals not defined in the kwargs")

        if n_ele > self.n_orbitals:
            raise TequilaException("number of electrons must not be greater than number of orbitlas")

        self.n_ele = n_ele

        self.parameter_map_ket = []
        for list in ket_fcircuit.variables:
            for x in list:
                self.parameter_map_ket.append(tq.assign_variable(x))
        # self.parameter_map_ket = [tq.assign_variable(x) for x in ket_fcircuit.variables]

        ket_instructions = ket_fcircuit.extract_indices()

        self.ket_generator = create_fermionic_generators(ket_instructions)

        bin_dict = generate_of_binary_dict(self.n_orbitals, self.n_ele // 2)

        self.ket = fqe.Wavefunction(param=[[self.n_ele, 0, self.n_orbitals]])  # probably only works for H

        if ket_fcircuit.initial_state is None:
            self.ket.set_wfn(strategy='hartree-fock')
        else:
            set_init_state(wfn = self.ket, n_ele=self.n_ele,n_orb=self.n_orbitals, init_state=ket_fcircuit.initial_state, bin_dict=bin_dict)


        bra_instructions = None
        init_bra = None
        if bra_fcircuit is not None:
            # self.parameter_map_bra = [tq.assign_variable(x[0]) for x in bra_fcircuit.variables]
            self.parameter_map_bra = []
            for list in bra_fcircuit.variables:
                for x in list:
                    self.parameter_map_bra.append(tq.assign_variable(x))

            bra_instructions = bra_fcircuit.extract_indices()
            self.bra_generator = create_fermionic_generators(bra_instructions)
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
        result = fqe.expectationValue(wfn=ket_t, ops=self.h_fqe, brawfn=bra_t)

        return result


    def print_ket(self):
        self.ket.print_wfn()

    def print_bra(self):
        self.bra.print_wfn()

    def print_ket_time_evolved(self):
        self.ket_time_evolved.print_wfn()

    def print_bra_time_evolved(self):
        self.bra_time_evolved.print_wfn()


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
    for idx, i in enumerate(wvf._state):
        if abs(i) > 1e-3: #todo check num of ones somewhere
            vec = bin(idx)[2:]
            if len(vec) < n_orb*2:
                vec = '0'*(n_orb*2-len(vec))+vec

            vec = vec[len(vec)//2:]
            indices.append(bin_dict[vec])
            values.append(abs(i)) #todo not sure about this


    return indices, values






