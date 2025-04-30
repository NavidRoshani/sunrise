import tequila as tq
import sunrise as sun
import numpy as np

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