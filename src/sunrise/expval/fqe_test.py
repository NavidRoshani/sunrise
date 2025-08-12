import sunrise as sun
import tequila as tq
import numpy as np
from fqe_expval import FQEBraKet
import matplotlib.pyplot as plt

mol = tq.Molecule("H 0 0 0\nH 0 0 1\nH 0 0 2\nH 0 0 3", "sto-3g").use_native_orbitals()
H = mol.make_hamiltonian()
c,h,g = mol.get_integrals()
# #
# #
# # # Single excitation
# # # (0,4): np.pi/2
# # angle = np.pi/2
# # print("teq:", tq.simulate(tq.ExpectationValue(mol.prepare_reference() + mol.make_excitation_gate((0,4), angle), H)))
# # # print(f"fqe:", FQEBraKet(h, g.elems, c, ket_instructions=[[indices]])({indices:angle}).real)
# # print("fqe:", FQEBraKet(one_body_integrals=h, two_body_integrals=g.elems, constant=c,n_ele=4, ket_instructions=[["a",(0,4)]])({"a":angle}).real)
# # print()
# # Single excitation
# # (0,4): np.pi/2
# angle = np.pi/2
#
# print("teq:", tq.simulate(tq.ExpectationValue(mol.prepare_reference() + mol.make_excitation_gate((5,1), angle), H)))
# print(tq.simulate(mol.prepare_reference() + mol.make_excitation_gate((1,5), angle)))
# # print(f"fqe:", FQEBraKet(h, g.elems, c, ket_instructions=[[indices]])({indices:angle}).real)
# print("fqe:", FQEBraKet(one_body_integrals=h, two_body_integrals=g.elems, constant=c,n_ele=4, ket_instructions=[["a",(1,5)]])({"a":-angle}).real)
# print()
#
# # Double excitation
# # (0,4,1,5): np.pi/2
# angle = 2
# print("new test")
# print("teq:", tq.simulate(tq.ExpectationValue(mol.prepare_reference() + mol.make_excitation_gate((0,4,1,5), angle), H)))
# print("teq:", tq.simulate(tq.ExpectationValue(mol.prepare_reference() + mol.make_excitation_gate([(0,4),(1,5)], angle), H)))
# print("fqe:", FQEBraKet(one_body_integrals=h, two_body_integrals=g.elems, constant=c, n_ele=4,
#                         ket_instructions=[["a",(0,4,1,5)]])({"a":angle}).real)
# fqe_l = []
# fqe_l2 = []
# tq_l=[]
# eee = FQEBraKet(one_body_integrals=h, two_body_integrals=g.elems, constant=c, n_ele=4,
#                         ket_instructions=[["a",(0,2,1,3)],["a",(4,6,5,7)]], init_ket="0101")
# pis = np.linspace(0,2*np.pi,100)
# # for i in pis:
# #
# #     fqe_l.append(eee({"a":i}).real)
# #     fqe_l2.append(FQEBraKet(one_body_integrals=h, two_body_integrals=g.elems, constant=c, n_ele=4,
# #                         ket_instructions=[["a",(0,2,1,3)],["a",(4,6,5,7)]], init_ket="0101")({"a":i}).real)
# #     tq_l.append(tq.simulate(tq.ExpectationValue(mol.prepare_reference() + mol.make_excitation_gate((0,2,1,3), angle)
# #                                                 + mol.make_excitation_gate((4,6,5,7), angle), H)))
# #
# #
# # plt.figure()
# # plt.plot(pis, fqe_l, label="FQE",linestyle="--")
# # plt.plot(pis, fqe_l2, label="FQE2",linestyle="-")
# # plt.plot(pis, tq_l, label="TQ", linestyle=":")
# # plt.legend()
# # plt.show()
# print()
#
# # Paired single excitation
# # [(0,4),(1,5)]: np.pi/2
# angle = np.pi/2
# print("teq:", tq.simulate(tq.ExpectationValue(mol.prepare_reference() + mol.make_excitation_gate((0,4), angle) + mol.make_excitation_gate((1,5), angle), H)))
# print("teq:", tq.simulate(tq.ExpectationValue(mol.prepare_reference() + mol.UR(0,2,angle), H)))
# print("fqe:", FQEBraKet(one_body_integrals=h, two_body_integrals=g.elems, constant=c,n_ele=4, ket_instructions=[["a",(0,4)],["a",(1,5)]])({"a":angle}).real)
# print()
#
# # Two single excitations with different angles
# # (0,4): np.pi/2, (1,5): np.pi/3
# angle1 = np.pi/2
# angle2 = np.pi/3
# print("teq:", tq.simulate(tq.ExpectationValue(mol.prepare_reference() + mol.make_excitation_gate((0,4), angle1) + mol.make_excitation_gate((1,5), angle2), H)))
# print("fqe:", FQEBraKet(one_body_integrals=h, two_body_integrals=g.elems, constant=c,n_ele=4,  ket_instructions=[["a",(0,4)],["b",(1,5)]])({"a":angle1, "b":angle2}).real)
# print()
#
# # Two double excitations with different angles
# # (0,4,1,5): np.pi/2, (0,6,1,7): np.pi/3
# angle1 = np.pi/2
# angle2 = np.pi/3
# print("teq:", tq.simulate(tq.ExpectationValue(mol.prepare_reference() + mol.make_excitation_gate((0,4,1,5), angle1) + mol.make_excitation_gate((0,6,1,7), angle2), H)))
# print("fqe:", FQEBraKet(one_body_integrals=h, two_body_integrals=g.elems, constant=c,n_ele=4,
#                         ket_instructions=[["a",(0,4,1,5)],["b",(0,6,1,7)]])({"a":angle1, "b":angle2}).real)
# print()
#
# Braket
# single and double
ket = mol.prepare_reference() + mol.make_excitation_gate((0,4), 2)
bra = mol.prepare_reference() + mol.make_excitation_gate((0,4,1,5),2)
print("teq:", tq.simulate(tq.BraKet(ket, bra, H)[0]))
print("fqe:", FQEBraKet(one_body_integrals=h, two_body_integrals=g.elems, constant=c, n_ele=4, init_ket="0011", init_bra="0011",
                        ket_instructions=[["a",(0,4)]], bra_instructions=[["b",(0,4,1,5)]]) ({"a":2,"b":2}).real)
#
# print()
# angle = np.pi/2
# print("teq:", tq.simulate(tq.ExpectationValue(mol.prepare_reference() + mol.make_excitation_gate((0,4), angle), H)))
# # print(f"fqe:", FQEBraKet(h, g.elems, c, ket_instructions=[[indices]])({indices:angle}).real)
# print("fqe:", FQEBraKet(one_body_integrals=h, two_body_integrals=g.elems, constant=c,n_ele=4,
#                         init_ket="0011",ket_instructions=[["a",(0,4)]])({"a":angle}).real)
# print()
#
#
# print()
# angle = 0
# print("teq:", tq.simulate(tq.ExpectationValue(mol.prepare_reference()+mol.make_excitation_gate((0,4,1,5),np.pi) +mol.make_excitation_gate((2,6,3,7),np.pi), H)))
# print( tq.simulate((mol.prepare_reference()+mol.make_excitation_gate((0,4,1,5),np.pi) +mol.make_excitation_gate((2,3,6,7),np.pi))))
# # print(f"fqe:", FQEBraKet(h, g.elems, c, ket_instructions=[[indices]])({indices:angle}).real)
# print("fqe:", FQEBraKet(one_body_integrals=h, two_body_integrals=g.elems, constant=c,n_ele=4,
#                         init_ket="1100",ket_instructions=[["a",(0,4)]])({"a":angle}).real)
# print()


print()
angle = np.pi/2
print("teq:", tq.simulate(tq.ExpectationValue(mol.prepare_reference()+mol.make_excitation_gate((0,4,1,5),np.pi) +mol.make_excitation_gate((2,6,3,7),np.pi)+ mol.make_excitation_gate((0,4), angle), H)))
print( tq.simulate((mol.prepare_reference()+mol.make_excitation_gate((0,4,1,5),np.pi) +mol.make_excitation_gate((2,3,6,7),np.pi)+ mol.make_excitation_gate((0,4), angle))))
# print(f"fqe:", FQEBraKet(h, g.elems, c, ket_instructions=[[indices]])({indices:angle}).real)
print("fqe:", FQEBraKet(one_body_integrals=h, two_body_integrals=g.elems, constant=c,n_ele=4,
                        init_ket="1100",ket_instructions=[["a",(0,4)]])({"a":angle}).real)
print()
