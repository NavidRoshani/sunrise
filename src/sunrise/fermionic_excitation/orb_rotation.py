import tequila as tq
from tequila import QCircuit,QTensor
from numpy import array,zeros,eye,tensordot,allclose

class OrbitalRotation():

    def __init__(self,orbitals:[int]=None,matrix:array=None):
        self.orbital = orbitals
        self.coeff = matrix #it will be changed to qtensor to allow variables
        assert len(self.orbital) == len(self.coeff)
        assert self.coeff.ndim == 2
        assert allclose(eye(len(self.coeff)), self.coeff.dot(self.coeff.T.conj()))
    def __add__(self, other):
        ndx = list(set(self.orbital+other.orbital))
        pos_idx = [ndx.index(i) for i in self.orbital]
        pos_jdx = [ndx.index(i) for i in other.orbital]
        new_a = eye(len(ndx))
        new_b = eye(len(ndx))
        new_a[:len(self.orbital), :len(self.orbital)] = self.coeff
        new_b[:len(other.orbital), :len(other.orbital)] = other.coeff
        ra = self._rot_mat(len(ndx), pos_idx)
        rb = self._rot_mat(len(ndx), pos_jdx)
        ap = ra.T.dot(new_a.dot(ra))
        bp = rb.T.dot(new_b.dot(rb))
        return OrbitalRotation(matrix=tensordot(ap,bp,axes=1),orbitals=ndx)
    def compile(self,**kwargs)->QCircuit:
        # TODO: fix line bellow properly
        U = tq.quantumchemistry.qc_base.QuantumChemistryBase.get_givens_circuit(unitary=self.coeff,**kwargs)
        U.map_qubits(qubit_map={i:idx for i,idx in enumerate(self.orbital)})
        return U
    def __eq__(self, other):
        return self.orbital==other.orbital and allclose(self.coeff,other.coeff)
    def _rot_mat(self,n: int, pos_dx):
        m = zeros(shape=(n, n))
        for i in range(len(pos_dx)):
            m[i, pos_dx[i]] = 1
        rest = [i for i in [*range(n)] if i not in pos_dx]
        for i in range(len(rest)):
            m[len(pos_dx) + i, rest[i]] = 1
        return m
    def __str__(self):
        pass
        # result = "circuit: \n"
        # for g in self.gates:
        #     result += str(g) + "\n"
        # return result
    def __repr__(self):
        return self.__str__()