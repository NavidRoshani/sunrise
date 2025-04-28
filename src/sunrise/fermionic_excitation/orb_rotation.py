import tequila as tq
from tequila import QCircuit,QTensor
from numpy import array,zeros
tq.gates.QubitExcitation()

class FermionicExcitation():

    def __init__(self,lfrom:[int]=None,lto:[int]=None,matrix:array=None):
        self.orbital = lfrom + lto
        self.coeff = matrix #it will be changed to qtensor to allow variables
        assert len(self.orbital) == len(self.coeff)
        assert self.coeff.ndim == 2
    def __add__(self, other):
        #it
        pass
    def compile(self)->QCircuit:
        pass