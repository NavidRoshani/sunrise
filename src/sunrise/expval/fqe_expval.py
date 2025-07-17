from sympy.physics.quantum.sho1d import Hamiltonian
from tequila import TequilaException
import tequila as tq
import numpy as np
from itertools import combinations
import openfermion
import fqe

import typing




class FQEBraKet():

    def __init__(self, n_ele, ket_instructions, one_body_integrals=None, two_body_integrals=None, constant=None,
                 bra_instructions = None, init_ket=None, init_bra=None):

        if (one_body_integrals is None and two_body_integrals is not None) \
                or one_body_integrals is not None and two_body_integrals is None:
            raise TequilaException("Both integrals are needed two conunstract a Hamiltonian")

        construct_ham= True
        if one_body_integrals is None and  two_body_integrals is None:
            construct_ham = False

        if construct_ham:
            h_of = make_fermionic_hamiltonian(one_body_integrals, two_body_integrals, constant,)
            self.h_fqe = fqe.get_hamiltonian_from_openfermion(h_of)
            self.n_orbitals = one_body_integrals.shape[0] #how to create this if its is non

        self.n_ele = n_ele
        self.init_ket = init_ket

        self.parameter_map_ket=[tq.assign_variable(x[0]) for x in ket_instructions]
        ket_instructions = [x[1:] for x in ket_instructions]
        self.ket_generator = create_fermionic_generators(ket_instructions)

        if bra_instructions is not None:
            self.parameter_map_bra = [tq.assign_variable(x[0]) for x in bra_instructions]
            bra_instructions = [x[1:] for x in bra_instructions]
            self.bra_generator = create_fermionic_generators(bra_instructions)

        self.bra_instructions = bra_instructions
        self.init_bra = init_bra
        self.bin_dict = generate_of_binary_dict(self.n_orbitals, self.n_ele // 2)



    def __call__(self,variables, *args, **kwargs):


        variables = tq.format_variable_dictionary(variables)

        ket = fqe.Wavefunction(param=[[self.n_ele, 0, self.n_orbitals]])  # probably only works for H

        if self.init_ket is None:
            ket.set_wfn(strategy='hartree-fock')
        else:

            i = self.bin_dict[self.init_ket]
            ket_coeff = ket.get_coeff((self.n_ele, 0))
            ket_coeff[i][i] = 1
            ket.set_wfn(strategy="from_data", raw_data={(self.n_ele, 0): ket_coeff})

        parameters_ket = [x(variables) for x in self.parameter_map_ket]
        zip_ket = zip(parameters_ket, self.ket_generator.values())
        for argument in zip_ket:
           ket = ket.time_evolve(0.5 * argument[0], argument[1])


        if self.bra_instructions is None:
            bra = None
        else:
            bra = fqe.Wavefunction(param=[[self.n_ele, 0, self.n_orbitals]])

            if self.init_bra is None:
                bra.set_wfn(strategy='hartree-fock')
            else:
                i = self.bin_dict[self.init_bra]
                bra_coeff = bra.get_coeff((self.n_ele, 0))
                bra_coeff[i][i] = 1
                bra.set_wfn(strategy="from_data", raw_data={(self.n_ele, 0): bra_coeff})

            parameters_bra = [x(variables) for x in self.parameter_map_bra]
            zip_bra = zip(parameters_bra, self.bra_generator.values())

            for argument in zip_bra:
                bra = bra.time_evolve(0.5 * argument[0], argument[1])


        result = fqe.expectationValue(wfn=ket, ops=self.h_fqe, brawfn=bra)

        return result

def make_fermionic_hamiltonian(one_body_integrals, two_body_integrals, constant, *args, **kwargs):

    one_body_coefficients, two_body_coefficients = openfermion.chem.molecular_data.spinorb_from_spatial(
        one_body_integrals, two_body_integrals)

    molecular_hamiltonian = openfermion.ops.representations.InteractionOperator(
        constant, one_body_coefficients, 1 / 2 * two_body_coefficients)
    fop = openfermion.transforms.get_fermion_operator(molecular_hamiltonian)

    return fop


def make_excitation_generator_op(indices: typing.Iterable[typing.Tuple[int, int]], form: str = 'fermionic',)\
                                    -> openfermion.FermionOperator:
    """
    Notes
    ----------
    Creates the transformed hermitian generator of UCC type unitaries:
          M(a^\dagger_{a_0} a_{i_0} a^\dagger{a_1}a_{i_1} ... - h.c.)
          where the qubit map M depends is self.transformation

    Parameters
    ----------
    indices : typing.Iterable[typing.Tuple[int, int]] :
        List of tuples [(a_0, i_0), (a_1, i_1), ... ] - recommended format, in spin-orbital notation (alpha odd numbers, beta even numbers)
        can also be given as one big list: [a_0, i_0, a_1, i_1 ...]
    form : str : (Default value None):
        Manipulate the generator to involution or projector
        set form='involution' or 'projector'
        the default is no manipulation which gives the standard fermionic excitation operator back
    Returns
    -------
    type
        1j*Transformed qubit excitation operator, depends on self.transformation
    """


    # check indices and convert to list of tuples if necessary
    if len(indices) == 0:
        raise TequilaException("make_excitation_operator: no indices given")
    elif not isinstance(indices[0], typing.Iterable):
        if len(indices) % 2 != 0:
            raise TequilaException("make_excitation_generator: unexpected input format of indices\n"
                                   "use list of tuples as [(a_0, i_0),(a_1, i_1) ...]\n"
                                   "or list as [a_0, i_0, a_1, i_1, ... ]\n"
                                   "you gave: {}".format(indices))
        converted = [(indices[2 * i], indices[2 * i + 1]) for i in range(len(indices) // 2)]
    else:
        converted = indices

    # convert everything to native python int
    # otherwise openfermion will complain
    converted = [(int(pair[0]), int(pair[1])) for pair in converted]

    # convert to openfermion input format
    ofi = []
    dag = []
    for pair in converted:
        assert (len(pair) == 2)
        ofi += [(int(pair[0]), 1),
                (int(pair[1]), 0)]  # openfermion does not take other types of integers like numpy.int64
        dag += [(int(pair[0]), 0), (int(pair[1]), 1)]

    op = openfermion.FermionOperator(tuple(ofi), 1.j)  # 1j makes it hermitian
    op += openfermion.FermionOperator(tuple(reversed(dag)), -1.j)

    if isinstance(form, str) and form.lower() != 'fermionic':
        # indices for all the Na operators
        Na = [x for pair in converted for x in [(pair[0], 1), (pair[0], 0)]]
        # indices for all the Ma operators (Ma = 1 - Na)
        Ma = [x for pair in converted for x in [(pair[0], 0), (pair[0], 1)]]
        # indices for all the Ni operators
        Ni = [x for pair in converted for x in [(pair[1], 1), (pair[1], 0)]]
        # indices for all the Mi operators
        Mi = [x for pair in converted for x in [(pair[1], 0), (pair[1], 1)]]

        # can gaussianize as projector or as involution (last is default)
        if form.lower() == "p+":
            op *= 0.5
            op += openfermion.FermionOperator(Na + Mi, 0.5)
            op += openfermion.FermionOperator(Ni + Ma, 0.5)
        elif form.lower() == "p-":
            op *= 0.5
            op += openfermion.FermionOperator(Na + Mi, -0.5)
            op += openfermion.FermionOperator(Ni + Ma, -0.5)

        elif form.lower() == "g+":
            op += openfermion.FermionOperator([], 1.0)  # Just for clarity will be subtracted anyway
            op += openfermion.FermionOperator(Na + Mi, -1.0)
            op += openfermion.FermionOperator(Ni + Ma, -1.0)
        elif form.lower() == "g-":
            op += openfermion.FermionOperator([], -1.0)  # Just for clarity will be subtracted anyway
            op += openfermion.FermionOperator(Na + Mi, 1.0)
            op += openfermion.FermionOperator(Ni + Ma, 1.0)
        elif form.lower() == "p0":
            # P0: we only construct P0 and don't keep the original generator
            op = openfermion.FermionOperator([], 1.0)  # Just for clarity will be subtracted anyway
            op += openfermion.FermionOperator(Na + Mi, -1.0)
            op += openfermion.FermionOperator(Ni + Ma, -1.0)
        else:
            raise TequilaException(
                "Unknown generator form {}, supported are G, P+, P-, G+, G- and P0".format(form))



    return op

def create_fermionic_generators(instructions):

    # instruction format [[(0,2,1,3),(2,4,3,5)],[(0,2),(1,3)]]
    #todo check for instruction format: even, repeating indeces
    generators={}

    for angle_idx, fermionic_group in enumerate(instructions):

        for x in fermionic_group:
            if len(fermionic_group[0]) != len(x):
                raise TequilaException("Trying to assign same angles to different type generators")
        # if ((len(fermionic_group[0]) != len(x)) for x in fermionic_group):
        #     raise TequilaException("Trying to assign same angles to different type generators") # needeed ?
        for fermionic_circuit in fermionic_group:
            if angle_idx in generators:
                generators[angle_idx] = generators[angle_idx] + make_excitation_generator_op(indices=fermionic_circuit)
            else:
                generators[angle_idx] = make_excitation_generator_op(indices=fermionic_circuit)

    return generators


def generate_of_binary_dict(n_orb: int, n_e: int) -> dict:

    """
    create a dictionary of all possibilities of a binary string of length n_orb with n_e number of 1s as a key and
    the integer value of the binary string as the value
    Parameters
    ----------
    n_orb: number of orbitals
    n_e: number of electrons

    Returns
    -------
    Dictionary with the key being the binary possibilities and the value being the integer value of the binary string
    """
    result = {}
    for index, positions in enumerate(combinations(range(n_orb), n_e)):
        s = ['0'] * n_orb
        for pos in positions:
            s[pos] = '1'
        binary_str = ''.join(s)

        result[binary_str[::-1]] = index
    return result



