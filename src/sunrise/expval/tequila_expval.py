import tequila as tq
from tequila import BraKet,QCircuit,QubitHamiltonian
from numbers import Number
from tequila import TequilaException


def TequilaBraket(U:[QCircuit,list,tuple]=None,bra:QCircuit=None,ket:QCircuit=None,reference:QCircuit=None,operator:QubitHamiltonian=None,*args,**kwargs):
    if 'circuit' in kwargs:
        circuit = kwargs['circuit']
        kwargs.pop('circuit')
        if U is not None:
            raise TequilaException('Two circuits provided?')
        else: U = circuit
    if U is not None and isinstance(U,(list,tuple)):
        if isinstance(U[0],Number):
            U = [[U,],]
        elif isinstance(U[0][0],Number):
            U = [U,]
        if "molecule" in kwargs:
            molecule = kwargs['molecule']
        else:
            nsos = max([max([max([idx[0] for idx in exct]+[idx[1] for idx in exct])]) for exct in U])
            geom = ''
            for i in range(nsos//2+1):
                geom += f'H 0. 0. {i}'
            molecule = tq.Molecule(geometry=geom,basis_set='sto-3g')
        Ux = QCircuit()
        for exct in U:
            Ux+= molecule.make_excitation_gate(indices=exct,angle=str(exct))
        U = Ux
    if U is not None and ket is not None:
        raise TequilaException('Two circuit provided?')
    elif U is not None:
        ket = U
    if reference is not None:
        ket = reference + ket
        if bra is not None:
            bra = reference + bra
        else:
            bra = reference
    if 'H' in kwargs:
        H = kwargs['H']
        kwargs.pop('H')
        if operator is not None:
            raise TequilaException('Two Operators provided?')
        else:
            operator = H
    return BraKet(ket=ket, bra=bra, operator=operator, *args, **kwargs)




