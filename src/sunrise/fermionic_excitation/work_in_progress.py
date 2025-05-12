import tequila as tq
import sunrise as sun
import numpy as np
from sunrise import OrbitalRotation as OR
# def rot_mat(n:int,pos_dx):
#     m = np.zeros(shape=(n,n))
#     for i in range(len(pos_dx)):
#         m[i,pos_dx[i]] = 1
#     rest = [i for i in [*range(n)] if i not in pos_dx]
#     print('REST ',rest)
#     for i in range(len(rest)):
#         m[len(pos_dx)+i,rest[i]] =1
#     return m
# a = np.array([[1,0],[1,0]])
# idx = [1,2]
# b = np.array([[0.1,0.2],[.3,0.4]])
# jdx = [2,3]
# print('original a\n',a)
# print('original b\n', b)
# ndx = list(set(idx+jdx))
# print('ndx ',ndx)
# pos_idx = [ndx.index(i) for i in idx]
# pos_jdx = [ndx.index(i) for i in jdx]
# print('pos_idx ',pos_idx)
# print('pos_jdx ',pos_jdx)
# new_a = np.eye(len(ndx))
# new_b = np.eye(len(ndx))
# new_a[:len(a),:len(a)]=a
# new_b[:len(b),:len(b)]=b
# # print('new_a\n',new_a)
# # print('new_b\n',new_b)
# ra = rot_mat(len(ndx),pos_idx)
# rb = rot_mat(len(ndx),pos_jdx)
# # print("rot_a:\n",ra)
# # print("rot_b:\n",rb)
# ap = ra.T.dot(new_a.dot(ra))
# bp = rb.T.dot(new_b.dot(rb))
# print("Final A\n",ap)
# print("Final B\n",bp)
# fin = np.tensordot(ap,bp,axes=1)
# print("Result\n",fin)

#a = tq.QTensor(shape=(3,3))
#b = np.eye(3)
#print(a)
#print(a.dot(b))
#print(np.dot(a,b))
#exit()
#a = tq.Variable("aaaa")
#print(isinstance(a,(tq.Objective,tq.Variable)))
#print(type(a))
#a = 2*tq.Variable("aaaa")
#print(isinstance(a,tq.Objective))
#print(type(a))
# if isinstance(a, tq.Variable) or abs(a) > 0.1:
#     print(a)
#exit()
# def f(x,y):
#     print('AAAAAAAA',x,y)
#     return x[0]+x[1]
# a = 1*tq.Variable("a")
# b = 1*tq.Variable('b')
# print('a: ',type(a))
# print('b: ',type(b))
# a.apply(np.cos)
# c = a+b
# print(type(c))
# print(c.transformation)
# c = b.apply(np.arctan)
# c = tq.Objective().binary_operator(left=a,right=b,op=np.arctan2)
# print(c.args)
# print(type(c.args[0]))
# d =c.wrap(op=f)
# print(d.args)
# print(vars(d))
# print(repr(tq.simulate(d,variables={"a":1,"b":1})))
# c.apply()
# exit()
# idx = [0,1]
# a = tq.QTensor(shape=(2,2),objective_list=np.zeros(shape=(2,2)).reshape(4))
# # print(repr(a))
# # exit()
# a[0,0] = tq.Variable("a00")
# a[0,1] = tq.Variable('a01')
# a[1,0] = tq.Variable('a10')
# a[1,1] = tq.Variable('a11')
# da = {'a00':0,'a01':1,'a10':1,'a11':0}
# # print(vars(a[1,0]),'--',type(a[1,0]))
# # print(vars(a[1,1]),'--',type(a[1,1]))
# # print(vars(a[0,0]),'--',type(a[0,0]))
# # print(len(a[0,0].args))
# # a = np.array([[0,1],[1,0]])
# b = tq.QTensor(shape=(2,2),objective_list=np.zeros(shape=(2,2)).reshape(4))
# b[0,0] = tq.Variable("b00")
# b[0,1] = tq.Variable("b00")
# b[1,0] = tq.Variable("b00")
# b[1,1] = tq.Variable("b00")
# da.update({"b00":1,"b01":-1,"b10":1,"b11":1})
# jdx = [1,2]
# b = (1/(np.sqrt(2)))*b
# print('Variables: ',da)
# RA = OR(orbitals=idx,matrix=a)
# # print("RA ",repr(RA.coeff))
# RB = OR(orbitals=jdx,matrix=b)
# # print("RB ",repr(RB.coeff))
# # print(repr(RA.coeff))
# # print(repr(tq.compile(RA.coeff,variables=da)))p
# # print(tq.simulate(tq.gates.X([0,1])+RA.compile(),variables={'a00':0,'a01':1,'a10':1,'a11':0}))
# # print(repr(tq.compile(RA.coeff,variables=da)))
# # print(RA.compile().map_variables({'a00':0,'a01':1,'a10':1,'a11':0}))
# RC = RA+RB
# # print(repr(RC.coeff))
# # print(repr(tq.simulate(RC.coeff,variables=da)))
# # print(RC.compile())
# # exit()
# # print('RC ',RC)
# RD = RB+RA
# # print('RD ',RD)
# print("RA ->",RA.compile())
# print("RB ->",RB.compile())
# print("RC ->",RC.compile())
# print("RD ->",RD.compile())
# RA = tq.gates.X([0,1])+RA.compile()
# RB = tq.gates.X([2,3])+RB.compile()
# RCc = tq.gates.X([0,1])+RC.compile() #first acts RA then RB, which is equal to just RB but with X([2,3])
# REc = tq.gates.X([2,3])+RC.compile() #RA reverses from X(2,3) to X(0,1) which makes RB unable to act
# RDc = tq.gates.X([0,1])+RD.compile() #RB cant act, only RA X(0,1) -> X(2,3)
# RFc = tq.gates.X([2,3])+RD.compile() #Fist acts RB and RA
# print('RA ',tq.simulate(objective=RA,variables=da))
# print('RB ',tq.simulate(objective=RB,variables=da))
# print('RC ',tq.simulate(objective=RCc,variables=da))
# print('RE ',tq.simulate(objective=REc,variables=da))
# print('RD ',tq.simulate(objective=RDc,variables=da))
# print('RF ',tq.simulate(objective=RFc,variables=da))
# RA  -1|0011>
# RB  0.5|001100> 0.5|000110> -0.5|001001> 0.5|000011>
# RC  -0.5|001100> -0.5|000110> 0.5|001001> -0.5|000011>
# RE  -1|110000>
# RD  -1|001100>
# RF  -0.5|110000> -0.5|010010> 0.5|100001> -0.5|000011>