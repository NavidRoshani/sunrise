import tequila as tq
from tequila import QCircuit,QTensor
from numpy import array,pad

class OrbitalRotation():

    def __init__(self,orbitals:[int]=None,matrix:array=None):
        self.orbital = orbitals
        self.coeff = matrix #it will be changed to qtensor to allow variables
        assert len(self.orbital) == len(self.coeff)
        assert self.coeff.ndim == 2
    def __add__(self, other):
        orb = self.orbital + other.orbital
        new_idx = [self.orbital]
        t = pad(self.orbital,[(0,len(other.orbital)),(0,len(other.orbital))])
        o = pad(other.orbital,[(len(self.orbital),0),(len(self.orbital),0)])
        return OrbitalRotation(matrix=t+o,orbitals=orb)
    def compile(self)->QCircuit:
        pass
