from tencirchem.static.ucc import *
from functools import partial
from collections import defaultdict
from time import time
from typing import Any, Tuple, Callable, List, Union

import numpy as np
from scipy.optimize import minimize
import pandas as pd
from pyscf.gto.mole import Mole
from pyscf.scf import RHF
import tensorcircuit as tc

from tencirchem.constants import DISCARD_EPS
from tencirchem.molecule import _Molecule
from tencirchem.utils.misc import reverse_qop_idx, scipy_opt_wrap, rdm_mo2ao, canonical_mo_coeff
from tencirchem.utils.circuit import get_circuit_dataframe
from tencirchem.static.engine_ucc import (
    get_civector,
    get_statevector,
    apply_excitation,
    translate_init_state,
    get_energy,get_energy_and_grad,
)
from tencirchem.static.ci_utils import get_ci_strings, get_ex_bitstring, get_addr, get_init_civector
from tencirchem.static.evolve_tensornetwork import get_circuit
from .engine_braket import get_expval,get_expval_and_grad
from tequila import Objective,Variable,QTensor

class EXPVAL(UCC):
    def __init__(
        self,
        mol: Union[Mole, RHF],
        init_method="mp2",
        active_space=None,
        aslst=None,  
        mo_coeff=None,
        mode="fermion",
        engine=None,
        run_hf=True,
        run_mp2=True,
        run_ccsd=True,
        run_fci=True, 
    ):
        r"""
        Initialize the class with molecular input.

        Parameters
        ----------
        mol: Mole or RHF
            The molecule as PySCF ``Mole`` object or the PySCF ``RHF`` object
        init_method: str, optional
            How to determine the initial amplitude guess. Accepts ``"mp2"`` (default), ``"ccsd"``, ``"fe"``
            and ``"zeros"``.
        active_space: Tuple[int, int], optional
            Active space approximation. The first integer is the number of electrons and the second integer is
            the number or spatial-orbitals. Defaults to None.
        aslst: List[int], optional
            Pick orbitals for the active space. Defaults to None which means the orbitals are sorted by energy.
            The orbital index is 0-based.

            .. note::
                See `PySCF document <https://pyscf.org/user/mcscf.html#picking-an-active-space>`_
                for choosing the active space orbitals. Here orbital index is 0-based, whereas in PySCF by default it
                is 1-based.
        mo_coeff: np.ndarray, optional
            Molecule coefficients. If provided then RHF is skipped.
            Can be used in combination with the ``init_state`` attribute.
            Defaults to None which means RHF orbitals are used.
        mode: str, optional
            How to deal with particle symmetry, such as whether force electrons to pair as hard-core boson (HCB).
            Possible values are ``"fermion"``, ``"qubit"`` and ``"hcb"``.
            Default to ``"fermion"``.
        engine: str, optional
            The engine to run the calculation. See :ref:`advanced:Engines` for details.
        run_hf: bool, optional
            Whether run HF for molecule orbitals. Defaults to ``True``.
            The argument has no effect if ``mol`` is a ``RHF`` object.
        run_mp2: bool, optional
            Whether run MP2 for initial guess and energy reference. Defaults to ``True``.
        run_ccsd: bool, optional
            Whether run CCSD for initial guess and energy reference. Defaults to ``True``.
        run_fci: bool, optional
            Whether run FCI  for energy reference. Defaults to ``True``.

        """
        super().__init__(mol=mol,
        init_method=init_method,
        active_space=active_space,
        aslst=aslst, #if breaking bcs here, try installing tcc-ng from github, not sure why it may install regular tcc, not ng
        mo_coeff=mo_coeff,
        mode=mode,
        engine=engine,
        run_hf=run_hf,
        run_mp2=run_mp2,
        run_ccsd=run_ccsd,
        run_fci=run_fci)
        
        # circuit related
        self._init_state_bra = None
        self._init_state_ket = None
        self.ex_ops_bra = None
        self.ex_ops_ket = None
        self._param_ids_bra = None
        self._param_ids_ket = None
        # self.init_guess_bra = None
        self.init_guess:dict = None #init_guess for al variables, type {"a":1,...}
        # optimization related
        self.scipy_minimize_options = None
        # optimization result
        self.opt_res = None
        # for manually set
        self._params_bra = None
        self._params_ket = None
        delattr(self,'_params')
        # delattr(self,'init_guess')
        delattr(self,'_param_ids')
        delattr(self,'_init_state')


    def get_opt_function(self, with_time: bool = False) -> Union[Callable, Tuple[Callable, float]]:
        """
        Returns the cost function in SciPy format for optimization.
        The gradient is included.
        Basically a wrapper to :func:`expval_and_grad`.

        Parameters
        ----------
        with_time: bool, optional
            Whether return staging time. Defaults to False.

        Returns
        -------
        opt_function: Callable
            The optimization cost function in SciPy format.
        time: float
            Staging time. Returned when ``with_time`` is set to ``True``.
        """
        expval_and_grad = scipy_opt_wrap(partial(self.expval_and_grad, engine=self.engine))

        time1 = time()
        if tc.backend.name == "jax":
            logger.info("JIT compiling the circuit")
            _ = expval_and_grad(np.zeros(self.n_params))
            logger.info("Circuit JIT compiled")
        time2 = time()
        if with_time:
            return expval_and_grad, time2 - time1
        return expval_and_grad
    

    def kernel(self) -> float:
        """
        The kernel to perform the VQE algorithm.
        The L-BFGS-B method in SciPy is used for optimization
        and configuration is possible by setting the ``self.scipy_minimize_options`` attribute.

        Returns
        -------
        e: float
            The optimized energy
        """
        assert len(self.param_ids_ket) == len(self.ex_ops_ket)
        assert len(self.param_ids_bra) == len(self.ex_ops_bra)
        #TODO: here I should check if all in bra==ket, and run super().kernel() will be faster and avoid problem two many variabls
        energy_and_grad, stating_time = self.get_opt_function(with_time=True)
        
        if self.init_guess is None:
            self.init_guess  = np.zeros(self.n_variables)
        # optimization options
        if self.scipy_minimize_options is None:
            # quite strict
            options = {"ftol": 1e1 * np.finfo(tc.rdtypestr).eps, "gtol": 1e2 * np.finfo(tc.rdtypestr).eps}
        else:
            options = self.scipy_minimize_options
        # x0 = [*self.init_guess_bra.copy()]
        # x1 = [*self.init_guess.copy()]
        # x0 = x0+x1
        print("N Variables ",self.n_variables)
        wrap_energy_and_grad = _EvalContainer(energy_and_grad,self.params)

        logger.info("Begin optimization")
        time1 = time()
        opt_res = minimize(wrap_energy_and_grad, x0=self.init_guess, jac=True, method="L-BFGS-B", options=options)
        time2 = time()

        if not opt_res.success:
            logger.warning("Optimization failed. See `.opt_res` for details.")

        opt_res["staging_time"] = stating_time
        opt_res["opt_time"] = time2 - time1
        opt_res["init_guess"] = self.init_guess
        # opt_res["init_guess_ket"] = self.init_guess
        # opt_res["init_guess_bra"] = self.init_guess_bra
        opt_res["e"] = float(opt_res.fun)
        self.opt_res = opt_res
        # prepare for future modification
        p = opt_res.x.copy()
        # opt_res.pop('x')
        # opt_res['x_bra']=p[:self.n_params_bra]
        # opt_res['x_ket']=p[self.n_params_bra:]

        self.params_bra = p#[:self.n_params_bra]
        self.params_ket = p#[self.n_params_bra:]
        return opt_res.e
    
    def expval(self, params_ket : Tensor = None, params_bra: Tensor = None, engine: str = None) -> float:
        """
        Evaluate the total Expectation Value.

        Parameters
        ----------
        params: Tensor, optional
            The circuit parameters. Defaults to None, which uses the optimized parameter
            and :func:`kernel` must be called before.
        engine: str, optional
            The engine to use. Defaults to ``None``, which uses ``self.engine``.

        Returns
        -------
        Expectation Value: float
            Total Expectation Value

        See Also
        --------
        civector: Get the configuration interaction (CI) vector.
        statevector: Evaluate the circuit state vector.
        expval_and_grad: Evaluate the total Expectation Value and parameter gradients.

        Examples
        --------
        >>> from tencirchem import UCCSD
        >>> from tencirchem.molecule import h2
        >>> uccsd = UCCSD(h2)
        >>> round(uccsd.energy([0, 0]), 8)  # HF state
        -1.11670614
        """
        self._sanity_check()
        if len(params_ket) == self.n_params and params_bra is None:
            params_bra = params_ket[:self.n_params_bra]
            params_ket = params_ket[self.n_params_bra:]
        if params_ket is None:
            params_ket = self.params_ket
        if params_bra is None:
            params_bra = self.params_bra
        params_ket  = self._check_params_argument(params_ket)
        params_bra = self._check_params_argument(params_bra)
        hamiltonian, _, engine = self._get_hamiltonian_and_core(engine)
        e = get_expval(hamiltonian=hamiltonian, n_qubits=self.n_qubits, n_elec_s=self.n_elec_s,engine= engine,mode=self.mode,
                       ex_ops=self.ex_ops_ket , ex_ops_bra=self.ex_ops_bra, 
                       params=self.params_ket ,params_bra=self.params_bra,
                       param_ids=self.param_ids_ket ,param_ids_bra=self.param_ids_bra,
                       init_state=self.init_state_ket , init_state_bra=self.init_state_bra)
        return float(e) + self.e_core
    
    def expval_and_grad(self, params_ket : Tensor = None, params_bra: Tensor = None, engine: str = None) -> Tuple[float, Tensor]:
        """
        Evaluate the total Expectation Value and parameter gradients.

        Parameters
        ----------
        params: Tensor, optional
            The circuit parameters. Defaults to None, which uses the optimized parameter
            and :func:`kernel` must be called before.
        engine: str, optional
            The engine to use. Defaults to ``None``, which uses ``self.engine``.

        Returns
        -------
        Expectation Value: float
            Total energy
        grad: Tensor
            The parameter gradients

        See Also
        --------
        civector: Get the configuration interaction (CI) vector.
        statevector: Evaluate the circuit state vector.
        Expectation Value: Evaluate the Operator Expectation Value.

        Examples
        --------
        >>> from tencirchem import UCCSD
        >>> from tencirchem.molecule import h2
        >>> uccsd = UCCSD(h2)
        >>> e, g = uccsd.energy_and_grad([0, 0])
        >>> round(e, 8)
        -1.11670614
        >>> g  # doctest:+ELLIPSIS
        array([..., ...])
        """
        self._sanity_check()
        if len(params_ket) == self.n_params and params_bra is None:
            params_bra = params_ket[:self.n_params_bra]
            params_ket = params_ket[self.n_params_bra:]
        if params_ket is None:
            params_ket = self.params_ket
        if params_bra is None:
            params_bra = self.params_bra
        params_ket  = self._check_params_argument(params_ket)
        params_bra = self._check_params_argument(params_bra)
        hamiltonian, _, engine = self._get_hamiltonian_and_core(engine)
        e, g = get_expval_and_grad(hamiltonian=hamiltonian, n_qubits=self.n_qubits, 
                                   n_elec_s=self.n_elec_s, engine=engine, mode=self.mode, 
                                   ex_ops=self.ex_ops_ket , ex_ops_bra=self.ex_ops_bra,
                                   params=params_ket ,params_bra=params_bra,
                                   param_ids=self.param_ids_ket, param_ids_bra=self.param_ids_bra,
                                   init_state=self.init_state_ket, init_state_bra= self.init_state_bra) 
        return float(e + self.e_core), tc.backend.numpy(g)

    def _check_params_argument(self, params, strict=True):
        if params is None:
            if self.params is not None:
                params = self.params
            else:
                if strict:
                    raise ValueError("Run the `.kernel` method to determine the parameters first")
                else:
                    if self.init_guess  is not None and self.init_guess_bra is not None:
                        params = self.init_guess_bra  + self.init_guess
                    else:
                        params = np.zeros(self.n_params)
        if len(params) != self.n_params_ket  and len(params) != self.n_params_bra and len(params) != self.n_params:
            raise ValueError(f"Incompatible parameter shape. {self.n_params} , {self.n_params_bra}  or {self.n_params_ket} is desired. Got {len(params)}")
        return tc.backend.convert_to_tensor(params).astype(tc.rdtypestr)
    
    def _sanity_check(self):
        if self.ex_ops_ket  is None or self.param_ids_ket is None:
            raise ValueError("`ex_ops_ket` or `param_ids_ket` not defined")
        if self.ex_ops_bra is None or self.param_ids_bra is None:
            raise ValueError("`ex_ops_bra` or `param_ids_bra` not defined")
        
        if self.param_ids_bra is not None and (len(self.ex_ops_bra) != len(self.param_ids_bra)):
            raise ValueError(
                f"Excitation operator size {len(self.ex_ops_bra)} and parameter size {len(self.param_ids_bra)} do not match"
            )
        if self.param_ids  is not None and (len(self.ex_ops_ket) != len(self.param_ids_ket)):
            raise ValueError(
                f"Excitation operator size {len(self.ex_ops_ket)} and parameter size {len(self.param_ids_ket)} do not match"
            )
    
    def civector(self, params: Tensor = None, engine: str = None,ket: bool = True) -> Tensor:
        """
        Evaluate the configuration interaction (CI) vector.

        Parameters
        ----------
        params: Tensor, optional
            The circuit parameters. Defaults to None, which uses the optimized parameter
            and :func:`kernel` must be called before.
        engine: str, optional
            The engine to use. Defaults to ``None``, which uses ``self.engine``.

        Returns
        -------
        civector: Tensor
            Corresponding CI vector

        See Also
        --------
        statevector: Evaluate the circuit state vector.
        energy: Evaluate the total energy.
        energy_and_grad: Evaluate the total energy and parameter gradients.

        Examples
        --------
        >>> from tencirchem import UCCSD
        >>> from tencirchem.molecule import h2
        >>> uccsd = UCCSD(h2)
        >>> uccsd.civector([0, 0])  # HF state
        array([1., 0., 0., 0.])
        """
        self._sanity_check()
        params = self._check_params_argument(params)
        self._check_engine(engine)
        if engine is None:
            engine = self.engine
        if ket:
            ex_ops  = self.ex_ops_ket
            param_ids = self.param_ids_ket 
            init_state = self.init_state_ket
        else:
            ex_ops  = self.ex_ops_bra
            param_ids = self.param_ids_bra
            init_state = self.init_state_bra
        civector = get_civector(
            params, self.n_qubits, self.n_elec_s, ex_ops, param_ids, self.mode, init_state, engine
        )
        return civector

    def _statevector_to_civector(self, statevector=None,ket: bool = True):
        if statevector is None:
            civector = self.civector(ket=ket)
        else:
            if len(statevector) == self.statevector_size:
                ci_strings = self.get_ci_strings()
                civector = statevector[ci_strings]
            else:
                if len(statevector) == self.civector_size:
                    civector = statevector
                else:
                    raise ValueError(f"Incompatible statevector size: {len(statevector)}")

        civector = tc.backend.numpy(tc.backend.convert_to_tensor(civector))
        return civector
    # since there's ci_vector method
    ci_strings = get_ci_strings

    def statevector(self, params: Tensor = None, engine: str = None, ket: bool = True) -> Tensor:
        """
        Evaluate the circuit state vector.

        Parameters
        ----------
        params: Tensor, optional
            The circuit parameters. Defaults to None, which uses the optimized parameter
            and :func:`kernel` must be called before.
        engine: str, optional
            The engine to use. Defaults to ``None``, which uses ``self.engine``.

        Returns
        -------
        statevector: Tensor
            Corresponding state vector

        See Also
        --------
        civector: Evaluate the configuration interaction (CI) vector.
        energy: Evaluate the total energy.
        energy_and_grad: Evaluate the total energy and parameter gradients.

        Examples
        --------
        >>> from tencirchem import UCCSD
        >>> from tencirchem.molecule import h2
        >>> uccsd = UCCSD(h2)
        >>> uccsd.statevector([0, 0])  # HF state
        array([0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        """
        self._sanity_check()
        params = self._check_params_argument(params)
        self._check_engine(engine)
        if engine is None:
            engine = self.engine
        if ket:
            ex_ops = self.ex_ops_ket
            param_ids = self.param_ids_ket 
            init_state = self.init_state_ket
        else:
            ex_ops = self.ex_ops_bra
            param_ids = self.param_ids_bra
            init_state = self.init_state_bra
        statevector = get_statevector(
            params, self.n_qubits, self.n_elec_s, ex_ops, param_ids, self.mode, init_state, engine
        )
        return statevector

    def get_circuit(self, params: Tensor = None, decompose_multicontrol: bool = False, trotter: bool = False, ket: bool = True) -> tc.Circuit:
        """
        Get the circuit as TensorCircuit ``Circuit`` object

        Parameters
        ----------
        params: Tensor, optional
            The circuit parameters. Defaults to None, which uses the optimized parameter.
            If :func:`kernel` is not called before, the initial guess is used.
        decompose_multicontrol: bool, optional
            Whether decompose the Multicontrol gate in the circuit into CNOT gates.
            Defaults to False.
        trotter: bool, optional
            Whether Trotterize the UCC factor into Pauli strings.
            Defaults to False.
        ket: bool, optional
            Whether to return ket or bra circuit
        Returns
        -------
        circuit: :class:`tc.Circuit`
            The quantum circuit.
        """
        if ket:
            ex_ops = self.ex_ops_ket
            param_ids = self.param_ids_ket 
            init_state = self.init_state_ket 
        else:
            ex_ops = self.ex_ops_bra
            param_ids = self.param_ids_bra
            init_state = self.init_state_bra
        if ex_ops is None:
            raise ValueError("Excitation operators not defined")
        params = self._check_params_argument(params, strict=False)
        return get_circuit(params,self.n_qubits,self.n_elec_s,ex_ops,param_ids,self.mode,init_state,decompose_multicontrol=decompose_multicontrol,trotter=trotter)

    def print_circuit(self,ket:bool=True):
        """
        Prints the circuit information. If you wish to print the circuit diagram,
        use :func:`get_circuit` and then call ``draw()`` such as ``print(ucc.get_circuit().draw())``.
        """
        c = self.get_circuit(ket=ket)
        df = get_circuit_dataframe(c)

        def format_flop(f):
            return f"{f:.3e}"

        formatters = {"flop": format_flop}
        print(df.to_string(index=False, formatters=formatters))

    def get_init_state_dataframe(self, init_state, coeff_epsilon: float = DISCARD_EPS,ket:bool=True) -> pd.DataFrame:
        """
        Returns initial state information dataframe.

        Parameters
        ----------
        coeff_epsilon: float, optional
            The threshold to screen out states with small coefficients.
            Defaults to 1e-12.

        Returns
        -------
        pd.DataFrame

        See Also
        --------
        init_state: The circuit initial state before applying the excitation operators.

        Examples
        --------
        >>> from tencirchem import UCC
        >>> from tencirchem.molecule import h2
        >>> ucc = UCC(h2)
        >>> ucc.init_state = [0.707, 0, 0, 0.707]
        >>> ucc.get_init_state_dataframe()   # doctest: +NORMALIZE_WHITESPACE
             configuration  coefficient
        0          0101        0.707
        1          1010        0.707
        """
        columns = ["configuration", "coefficient"]
        if init_state is None:
            if ket:
                if self.init_state_ket is None:
                    init_state = get_init_civector(self.civector_size)
                else:
                    init_state = self.init_state_ket
            else:
                if self.init_state_bra is None:
                    init_state = get_init_civector(self.civector_size)
                else:
                    init_state = self.init_state_bra
        ci_strings = self.get_ci_strings()
        ci_coeffs = translate_init_state(init_state, self.n_qubits, ci_strings)
        data_list = []
        for ci_string, coeff in zip(ci_strings, ci_coeffs):
            if np.abs(coeff) < coeff_epsilon:
                continue
            ci_string = bin(ci_string)[2:]
            ci_string = "0" * (self.n_qubits - len(ci_string)) + ci_string
            data_list.append((ci_string, coeff))
        return pd.DataFrame(data_list, columns=columns)

    def get_excitation_dataframe(self,ket:bool=True) -> pd.DataFrame:
        '''
        ket:bool Whether to refered to the ket (True) or bra (False)
        '''
        columns = ["excitation", "configuration", "parameter", "initial guess"]
        if ket:
            if self.ex_ops_ket  is None:
                return pd.DataFrame(columns=columns)

            if self.params_ket  is None:
                # optimization not done
                params = [None] * len(self.init_guess)
            else:
                params = self.params_ket 

            if self.param_ids_ket is None:
                # see self.n_params_ket
                param_ids = range(len(self.ex_ops_ket))
            else:
                param_ids = self.param_ids_ket
            ex_ops =self.ex_ops_ket
            init_guess = self.init_guess 
        else:
            if self.ex_ops_bra is None:
                return pd.DataFrame(columns=columns)

            if self.params_bra is None:
                # optimization not done
                params = [None] * len(self.init_guess_bra)
            else:
                params = self.params_bra

            if self.param_ids is None:
                # see self.n_params
                param_ids = range(len(self.ex_ops_bra))
            else:
                param_ids = self.param_ids_bra
            ex_ops =self.ex_ops_bra
            init_guess = self.init_guess_bra
            
        data_list = []

        for i, ex_op in zip(param_ids, ex_ops):
            bitstring = get_ex_bitstring(self.n_qubits, self.n_elec_s, ex_op, self.mode)
            data_list.append((ex_op, bitstring, params[i], init_guess[i]))
        return pd.DataFrame(data_list, columns=columns)

    def print_excitations(self):
        print('Bra: ',self.get_excitation_dataframe(ket=False).to_string())
        print('Ket: ',self.get_excitation_dataframe(ket=True).to_string())

    def print_ansatz(self):
        df_dict = {
            "#qubits": [self.n_qubits],
            "#params_ket": [self.n_params_ket],
            "#params_bra": [self.n_params_bra],
            "#excitations_ket": [len(self.ex_ops_ket)],
            "#excitations_bra": [len(self.ex_ops_bra)],
        }
        if self.init_state_bra is None:
            df_dict["initial condition bra"] = "RHF"
        else:
            df_dict["initial condition bra"] = "custom"

        if self.init_state is None:
            df_dict["initial condition ket"] = "RHF"
        else:
            df_dict["initial condition ket"] = "custom"
        print(pd.DataFrame(df_dict).to_string(index=False))

    def print_init_state_bra(self):
        print('Bra: ',self.get_init_state_dataframe(self.init_state_bra).to_string())
    
    def print_init_state_ket(self):
        print('Ket: ',self.get_init_state_dataframe(self.init_state_ket).to_string())

    def print_init_state(self):
        self.print_init_state_bra()
        self.print_init_state_ket()
    
    def print_summary(self, include_circuit: bool = False):
        """
        Print a summary of the class.

        Parameters
        ----------
        include_circuit: bool
            Whether include the circuit section.

        """
        print("################################ Ansatz ###############################")
        self.print_ansatz()
        if self.init_state_bra is not None or self.init_state_ket is not None:
            print("############################ Initial Condition ########################")
            if self.init_state_bra is not None:
                self.print_init_state_bra()
            if self.init_state_ket is not None:
                self.print_init_state_ket()
        if include_circuit:
            print("############################### Circuit ###############################")
            self.print_circuit()
        print("############################### Energy ################################")
        self.print_energy()
        print("############################# Excitations #############################")
        self.print_excitations()
        print("######################### Optimization Result #########################")
        if self.opt_res is None:
            print("Optimization not run (.opt_res is None)")
        else:
            print(self.opt_res)

    @property
    def init_state_bra(self) -> Tensor:
        """
        The circuit initial state before applying the excitation operators. Usually RHF.

        See Also
        --------
        get_init_state_dataframe: Returns initial state information dataframe.
        """
        return self._init_state_bra
    
    @property
    def init_state_ket(self) -> Tensor:
        """
        The circuit initial state before applying the excitation operators. Usually RHF.

        See Also
        --------
        get_init_state_dataframe: Returns initial state information dataframe.
        """
        return self._init_state_ket
    
    @property
    def init_state(self) -> Tensor:
        """
        The circuit initial state before applying the excitation operators. Usually RHF.

        See Also
        --------
        get_init_state_dataframe: Returns initial state information dataframe.
        """
        return self._init_state_bra,self._init_state_ket

    @init_state_bra.setter
    def init_state_bra(self, init_state_bra):
        self._init_state_bra = init_state_bra
    
    @init_state_ket.setter
    def init_state_ket(self, init_state_ket):
        self._init_state_ket = init_state_ket
    
    @init_state.setter
    def init_state(self, init_state):
        self._init_state_bra = init_state
        self._init_state_ket = init_state

    @property
    def ex_ops(self) -> Tensor:
        """
        Excitation operators applied to the bra + ket.
        """
        return self.ex_ops_bra + self.ex_ops_ket 

    @ex_ops.setter
    def ex_ops(self, ex_ops):
        self.ex_ops_bra = ex_ops
        self.ex_ops_ket = ex_ops

    @property
    def params_bra(self) -> Tensor:
        """The circuit parameters."""
        if self._params_bra is not None:
            return self._params_bra
        elif self.opt_res is not None:
            return self.opt_res.x
        return None
    
    @property
    def params_ket(self) -> Tensor:
        """The circuit parameters."""
        if self._params_ket is not None:
            return self._params_ket
        elif self.opt_res is not None:
            return self.opt_res.x
        return None
    
    @property
    def params(self) -> Tensor:
        """The circuit parameters."""
        return self.params_bra + self.params_ket
    
    @params_bra.setter
    def params_bra(self, params_bra):
        # params_bra = [Variable(i) for i in params_bra]
        self._params_bra = params_bra

    @params_ket.setter
    def params_ket(self, params_ket):
        # params_ket = [Variable(i) for i in params_ket]
        self._params_ket = params_ket
        if self.params_bra is None:
            self._params_bra = params_ket
    
    @params.setter
    def params(self, params):
        self.params_bra = params
        self.params_ket = params

    @property
    def param_ids_bra(self) -> List[int]:
        """The mapping from excitations operators to parameters."""
        if self._param_ids_bra is None:
            if self.ex_ops_bra is None:
                raise ValueError("Excitation operators BRA not defined")
            else:
                return tuple(range(len(self.ex_ops_bra)))
        return self._param_ids_bra
    
    @property
    def param_ids_ket(self) -> List[int]:
        """The mapping from excitations operators to parameters."""
        if self._param_ids_ket is None:
            if self.ex_ops_ket is None:
                raise ValueError("Excitation operators KET not defined")
            else:
                c = 0 if self.ex_ops_bra is None else len(self.ex_ops_bra)
                return tuple(range(c,len(self.ex_ops_ket)+c))
        return self._param_ids_ket

    @property
    def param_ids(self) -> List[int]:
        """The mapping from excitations operators to parameters."""
        return self.param_ids_bra + self.param_ids_ket

    @param_ids_bra.setter
    def param_ids_bra(self, v):
        self._param_ids_bra = v
        if self._param_ids_ket is None:
            self._param_ids_ket = v
    
    @param_ids_ket.setter
    def param_ids_ket(self, v):
        self._param_ids_ket = v
        if self._param_ids_bra is None:
            self._param_ids_bra = v
    
    @param_ids.setter
    def param_ids(self, v):
        self._param_ids_bra = v
        self._param_ids_ket = v

    @property 
    def param_to_ex_ops_bra(self):
        d = defaultdict(list)
        for i, j in enumerate(self.param_ids_bra):
            d[j].append(self.ex_ops_bra[i])
        return d
    
    @property
    def param_to_ex_ops_ket(self):
        d = defaultdict(list)
        c = self.n_params_bra
        for i, j in enumerate(self.param_ids_ket):
            d[j+c].append(self.ex_ops_ket[i])
        return d
    
    @property
    def param_to_ex_ops(self):
        d = self.param_to_ex_ops_bra
        for i,j in enumerate(self.param_ids_ket):
            d[j+self.n_params_bra].append(self.ex_ops_ket[i])
        return d 

    @property
    def n_params(self) -> int:
        """The number of parameter in the ansatz/circuit."""
        # this definition ensures that `param[param_id]` is always valid
        return self.n_params_bra + self.n_params_ket
    
    @property
    def n_params_bra(self) -> int:
        """The number of parameter in the ansatz/circuit."""
        # this definition ensures that `param[param_id]` is always valid
        if not self.param_ids_bra:
            return 0
        return max(self.param_ids_bra) + 1
    
    @property
    def n_params_ket(self) -> int:
        """The number of parameter in the ansatz/circuit."""
        # this definition ensures that `param[param_id]` is always valid
        if not self.param_ids_ket:
            return 0
        return max(self.param_ids_ket) + 1
    
    @property
    def n_variables(self) -> int:
        return len(Objective(self.params).extract_variables())
    
    @property
    def n_variables_bra(self) -> int:
        return len(Objective(self.params_bra).extract_variables())
                   
    @property
    def n_variables_ket(self) -> int:
        return len(Objective(self.params_ket).extract_variables())

    @property
    def variables(self):
        pass

class _EvalContainer:
    """
    Container Class to access scipy and keep the optimization history.
    This class is used by the SciPy optimizer and should not be used elsewhere.

    Attributes
    ---------
    objective:
        the objective to evaluate.
    param_keys:
        the dictionary mapping parameter keys to positions in a numpy array.
    samples:
        the number of samples to evaluate objective with.
    save_history:
        whether or not to save, in a history, information about each time __call__ occurs.
    print_level
        dictates the verbosity of printing during call.
    N:
        the length of param_keys.
    history:
        if save_history, a list of energies received from every __call__
    history_angles:
        if save_history, a list of angles sent to __call__.

    Class copypasted from Tequila, saludos.
    """

    def __init__(
        self, fun, param_keys
    ):
        self.fun = fun
        self.param_keys = param_keys
        self.params = Objective(param_keys).extract_variables()
        print(self.params)
        print(self.param_keys)
        self.N = len(self.params)

    def __call__(self, p, *args, **kwargs):
        """
        call a wrapped objective.
        Parameters
        ----------
        p: numpy array:
            Parameters with which to call the Function.
        args
        kwargs

        Returns
        -------
        numpy.array:
            value of self.objective with p translated into variables, as a numpy array.
        """
        angles = dict((self.params[i], p[i]) for i in range(self.N))
        variabes = [v(angles) for v in self.param_keys]

        values = self.fun(variabes)
        result = [values[0]]
        gradient = {}
        for k in range(len(self.params)):
            var = self.param_keys[k]
            if var not in gradient:
                gradient[var]=0.0
            
            gradient[var] += values[1][k]

        result += list(gradient.values())


        return result
