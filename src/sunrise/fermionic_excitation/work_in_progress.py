import tequila as tq
import sunrise as sun
import numpy as np

a = np.array([[0,1],[2,3]])
idx = [1,2]
b = np.array([[0,1],[1,0]])
jdx = [3,4]
print('original a\n',a)
print('original b\n', b)
ndx = list(set(idx+jdx))
print('ndx ',ndx)

pos_idx = [ndx.index(i) for i in idx]
pos_jdx = [ndx.index(i) for i in jdx]
print('pos_idx ',pos_idx)
print('pos_jdx ',pos_jdx)
r_idx = [(idx[i],pos_idx[i]) for  i in range(len(idx))]
r_jdx = [(jdx[i],pos_jdx[i]) for  i in range(len(jdx))]
print("R idx ",r_idx)
print("R jdx ",r_jdx)
new_a = np.eye(len(ndx))
new_b = np.eye(len(ndx))
new_a[:len(a),:len(a)]=a
new_b[:len(b),:len(b)]=b
print('new_a\n',new_a)
print('new_b\n',new_b)
ra = np.zeros(shape=new_a.shape)
rb = np.zeros(shape=new_b.shape)