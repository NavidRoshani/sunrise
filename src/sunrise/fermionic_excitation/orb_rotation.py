import numpy as np
import tequila as tq
from tequila import QCircuit,QTensor
from numpy import zeros,eye,allclose,dot,ndarray,array

class OrbitalRotation():

    def __init__(self,orbitals:[int]=None,matrix:[QTensor,ndarray,list]=None):
        if isinstance(matrix,list):
            matrix = array(list)
        if isinstance(matrix,ndarray) and not isinstance(matrix,QTensor):
            assert allclose(eye(len(matrix)), matrix.dot(matrix.T.conj())) #normalized
            matrix = QTensor(shape=matrix.shape,objective_list=matrix.reshape(matrix.shape[0]*matrix.shape[1]))
        self.orbital = orbitals
        self.coeff = matrix
        assert len(self.orbital) == len(self.coeff)
        assert self.coeff.ndim == 2
        assert self.coeff.shape[0] == self.coeff.shape[1] #is square
        # assert allclose(eye(len(self.coeff)), self.coeff.dot(self.coeff.T.conj())) #not normalization can be enforced, let to the user
    def __add__(self, other):
        ndx = list(set(self.orbital+other.orbital))
        pos_idx = [ndx.index(i) for i in self.orbital]
        pos_jdx = [ndx.index(i) for i in other.orbital]
        new_a = self.pad_eye(self.coeff,size=len(ndx))
        new_b = self.pad_eye(other.coeff,size=len(ndx))
        ra = self._rot_mat(len(ndx), pos_idx)
        rb = self._rot_mat(len(ndx), pos_jdx)
        ap = ra.T.dot(new_a.dot(ra))
        bp = rb.T.dot(new_b.dot(rb))
        return OrbitalRotation(matrix=bp.dot(ap),orbitals=ndx)
    def compile(self,**kwargs)->QCircuit:
        geom = ""
        for k in range(2*(len(self.coeff)//2+1)):
            geom += f"h 0.0 0.0 {1.5 * k}\n"
        dummy = tq.Molecule(geometry=geom,basis_set='sto-3g')
        U = dummy.get_givens_circuit(unitary=self.coeff,**kwargs)
        d = {2*i:2*idx for i,idx in enumerate(self.orbital)}
        d.update({2*i+1:2*idx+1 for i,idx in enumerate(self.orbital)})
        U = U.map_qubits(qubit_map=d)
        return U
    # def __eq__(self, other):
    #     return self.orbital==other.orbital and allclose(self.coeff,other.coeff)
    def _rot_mat(self,n: int, pos_dx):
        m = QTensor(shape=(n,n),objective_list=zeros(shape=(n,n)).reshape(n*n))
        for i in range(len(pos_dx)):
            m[i, pos_dx[i]] = 1
        rest = [i for i in [*range(n)] if i not in pos_dx]
        for i in range(len(rest)):
            m[len(pos_dx) + i, rest[i]] = 1
        return m
    def __str__(self):
        result = 'Orbital Rotation \n'
        result += f'acting on MO: {self.orbital}\n'
        result += f'with matrix: \n {self.coeff}'
        return result
    def __repr__(self):
        return self.__str__()
    def pad_eye(self,matrix:QTensor,size)->QTensor:
        '''
        Increases the square Qtensor size to (size,size) setting new diag entries to one.
        Useful for Qtensor  tensor product.
        '''
        new = QTensor(shape=(size,size),objective_list=eye(size).reshape(size**2))
        for i in range(len(matrix)):
            for j in range(i,len(matrix)):
                new[i,j] = matrix[i,j]
                new[j,i] = matrix[j,i]
        return new