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
        ov = [i for i in self.orbital if i in other.orbital]
        if len(ov):
            pass #how to carefully consider when overlap. i.e. [0,2] A + [0,4] B
        else:
            orb = self.orbital + other.orbital
            t = pad(self.orbital,[(0,len(other.orbital)),(0,len(other.orbital))])
            o = pad(other.orbital,[(len(self.orbital),0),(len(self.orbital),0)])
            return OrbitalRotation(matrix=t+o,orbitals=orb)
    def compile(self)->QCircuit:
        pass
    