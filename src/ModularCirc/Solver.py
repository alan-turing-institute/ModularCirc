from .Time import TimeClass
from .StateVariable import StateVariable
from .Models.OdeModel import OdeModel
from .HelperRoutines import bold_text
from .Models.OdeModel import OdeModel

import pandas as pd
import numpy as np
import numba as nb

from scipy.integrate import solve_ivp
from scipy.linalg import solve
from scipy.optimize import newton, approx_fprime, root, least_squares

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee
from scipy.linalg import bandwidth
from scipy.integrate import LSODA

import warnings

class Solver():
    def __init__(self,
                model:OdeModel=None,
                 ) -> None:

        self.model = model


        # Primary State Variables: Primary state variables are the main variables that are directly integrated over time
        # using their respective differential equations. These variables are updated using their time derivatives (dudt_func).
        # They are essential for the system's dynamics and are typically the focus of the numerical integration process.

        # Secondary State Variables: Secondary state variables are derived from the primary state variables. They do not
        # have their own differential equations but are instead computed from the primary state variables using algebraic
        # relationships (u_func). These variables are updated based on the current values of the primary state variables.

        # DataFrame containing all state variable data from the model.
        self._asd = model.all_sv_data

        # Dictionary of state variables from the model.
        self._vd  = model._state_variable_dict

        # Time object from the model
        self._to  = model.time_object


        # Global variables for the solver

        # Dictionary to store update functions for primary state variables.
        self._global_psv_update_fun  = {}
        # Dictionary to store update functions for secondary state variables.
        self._global_ssv_update_fun  = {}

        # Dictionary to store the names of the update functions for primary state variables.
        self._global_psv_update_fun_n  = {}
        # Dictionary to store the names of the update functions for secondary state variables.
        self._global_ssv_update_fun_n  = {}

        # Dictionary to store the indexes of the primary state variables.
        self._global_psv_update_ind  = {}
        # Dictionary to store the indexes of the secondary state variables.
        self._global_ssv_update_ind  = {}

        # List to store the names of the primary state variables.
        self._global_psv_names      = []
        # Dictionary to store the names of the secondary state variables.
        self._global_sv_init_fun    = {}

        # Dictionary to store the indexes of the secondary state variables.
        self._global_sv_init_ind    = {}

        # Dictionary mapping the state variable names to their indexes.
        self._global_sv_id          = {key: id   for id, key in enumerate(model.all_sv_data.columns.to_list())}
        # Dictionary mapping the indexes to the state variable names.
        self._global_sv_id_rev      = {id: key   for id, key in enumerate(model.all_sv_data.columns.to_list())}

        # Series to store state variables initialized by functions.
        self._initialize_by_function = pd.Series()

        # Number of sub-iterations for the solver. <- is this right? LB.
        self._N_sv = len(self._global_sv_id)

        # Variable to store the number of converged cycles.
        self._Nconv = None

        # Number of sub-iterations for the solver.
        self._n_sub_iter = 1

        # flag for checking if the model is converged or not...
        self.converged = False


    def setup(self,
              optimize_secondary_sv:bool=False,
              suppress_output:bool=False,
              step_tol:float=1e-2,
              conv_cols:list=None,
              method:str='BDF',
              atol=1e-6,
              rtol=1e-6,
              step = 1,
              )->None:
        """
        Method for detecting which are the principal variables and which are the secondary ones.

        ## Inputs
        optimize_secondary_sv : boolean
            flag used to switch on the optimization for secondary variable computations, this flag needs to be
            true when not all of the secondary variables can be expressed in terms of primary variables.
        """
        self._optimize_secondary_sv = optimize_secondary_sv
        self._step_tol  = step_tol
        self._conv_cols = conv_cols
        self._method    = method
        self._atol      = atol
        self._rtol      = rtol
        self.step       = step


        # Loop over the state variables and check if they have an update function,
        # This code ensures that each state variable's update function is correctly assigned and indexed,
        # allowing the solver to update the state variables during the simulation
        for key, component in self._vd.items():

            # Get the index of the state variable.
            mkey = self._global_sv_id[key] # _global_sv_id - maps the state variable names to their indexes.

            # initialization function for a state variable. This function is used to initialize the state
            # variable at the beginning of the simulation.
            if component.i_func is not None:
                if not suppress_output: print(f" -- Variable {bold_text(key)} added to the init list.")
                if not suppress_output: print(f'    - name of update function: {bold_text(component.i_name)}')
                if not suppress_output: print(f'    - inputs: {component.i_inputs.to_list()}')
                self._initialize_by_function[key] = component
                self._global_sv_init_fun[mkey] = component.i_func
                self._global_sv_init_ind[mkey] = [self._global_sv_id[key2] for key2 in component.i_inputs.to_list()]

            # derivative function for a state variable. This function is used to update the state variable
            # during the numerical integration process.
            if component.dudt_func is not None:
                if not suppress_output: print(f" -- Variable {bold_text(key)} added to the principal variable key list.")
                if not suppress_output: print(f'    - name of update function: {bold_text(component.dudt_name)}')
                if not suppress_output: print(f'    - inputs: {component.inputs.to_list()}')
                self._global_psv_update_fun[mkey]   = component.dudt_func
                self._global_psv_update_fun_n[mkey] = component.dudt_name
                self._global_psv_update_ind[mkey]   = [self._global_sv_id[key2] for key2 in component.inputs.to_list()]

                # Pad the index array to the length of the state variable array.
                self._global_psv_update_ind[mkey]   = np.pad(self._global_psv_update_ind[mkey],
                                                             (0, self._N_sv-len(self._global_psv_update_ind[mkey])),
                                                             mode='constant', constant_values=-1)

                # Add the state variable name to the global primary state variable list.
                self._global_psv_names.append(key)

            # updated function for the secondary state variable. This function is used to update the state variable
            # based on the current values of the primary state variables using algebraic relationships.
            elif component.u_func is not None:
                if not suppress_output: print(f" -- Variable {bold_text(key)} added to the secondary variable key list.")
                if not suppress_output: print(f'    - name of update function: {bold_text(component.u_name)}')
                if not suppress_output: print(f'    - inputs: {component.inputs.to_list()}')
                self._global_ssv_update_fun[mkey]   = component.u_func
                self._global_ssv_update_fun_n[mkey] = component.u_name
                self._global_ssv_update_ind[mkey]   = [self._global_sv_id[key2] for key2 in component.inputs.to_list()]
                self._global_ssv_update_ind[mkey]   = np.pad(self._global_ssv_update_ind[mkey],
                                                             (0, self._N_sv-len(self._global_ssv_update_ind[mkey])),
                                                             mode='constant', constant_values=-1)
            else:
                continue

        if not suppress_output: print(' ')

        self.generate_dfdt_functions()


        if self._conv_cols is None:
            # If no specific columns for convergence (_conv_cols) are provided,
            # automatically select columns from the DataFrame (_asd) whose names
            # contain 'v_' or 'p_', as variables of interest for convergence checks.
            self._cols = [col for col in self._asd.columns if 'v_' in col or 'p_' in col]
        else:
            # If specific convergence columns are provided, use them directly.
            self._cols = self._conv_cols

        # End the method without returning any specific value.
        return None


    def generate_dfdt_functions(self):

        """ Generating the functions needed to compute the derivatives of the state variables over time. These functions are
        used during the numerical integration process to update the state variables."""

        # Extract function arrays and indices for class attribute storage
        funcs1 = list(self._global_sv_init_fun.values())
        ids1   = list(self._global_sv_init_ind.values())
        funcs2 = np.array(list(self._global_ssv_update_fun.values()))
        ids2   = np.stack(list(self._global_ssv_update_ind.values()))
        keys3  = np.array(list(self._global_psv_update_fun.keys()))
        keys4  = np.array(list(self._global_ssv_update_fun.keys()))
        funcs3 = np.array(list(self._global_psv_update_fun.values()))
        ids3   = np.stack(list(self._global_psv_update_ind.values()))

        T = self._to.tcycle
        N_zeros_0 = len(self._global_sv_id)
        _n_sub_iter = self._n_sub_iter
        _optimize_secondary_sv = self._optimize_secondary_sv

        # stores the dependencies of primary variables
        keys3_dict = dict()
        for key, line in zip(keys3,ids3):
            line2= [val for val in np.unique(line) if val != -1]
            keys3_dict[key] = set(line2)

        keys3_back_dict = dict()
        for key, val in enumerate(keys3_dict):
            keys3_back_dict[val] = key

        # stores the dependencies of secondary variables
        keys4_dict = dict()
        for key, line in zip(keys4,ids2):
            line2= [val for val in np.unique(line) if val != -1]
            keys4_dict[key] = set(line2)

        #  combines dependencies to create a sparsity map.
        keys3_dict2 = dict()
        for key in keys3_dict.keys():
            keys3_dict2[key] = set()
            for val in keys3_dict[key]:
                if val in keys3:
                    keys3_dict2[key].update({val,})
                else:
                    keys3_dict2[key].update(keys4_dict[val])

        sparsity_map = dict()

        for i, key in enumerate(keys3):
            sparsity_map[i] = set()
            for val in keys3_dict2[key]:
                sparsity_map[i].add(keys3_back_dict[val])

        # creates a sparse matrix from the sparsity map
        mat = np.zeros((len(sparsity_map),len(sparsity_map)))
        for key, rows in sparsity_map.items():
            mat[key, np.array(list(rows), dtype=np.int64)] = 1

        # uses the reverse cuthill mckee algorithm to reduce the bandwidth of the matrix
        sparse_mat = csr_matrix(mat)
        perm = reverse_cuthill_mckee(sparse_mat, symmetric_mode=False)
        
        # Store permutation as indices instead of dense matrix for better performance
        self.perm_indices = perm.astype(np.int32)
        self.inv_perm_indices = np.argsort(perm).astype(np.int32)

        # reorders the sparse matrix to reduce the bandwidth
        sparse_mat_reordered = sparse_mat[perm, :][:, perm]

        # calculates the bandwidth of the reordered matrix
        sparse_mat_reordered_indexes = np.argwhere(sparse_mat_reordered.toarray())
        temp = sparse_mat_reordered_indexes[:,0] - sparse_mat_reordered_indexes[:,1]
        uband = np.abs(np.min(temp))
        lband = np.max(temp)

        self.lband = lband
        self.uband = uband

        # Store function arrays and indices as class attributes
        self._funcs1 = list(funcs1)
        self._ids1 = list(ids1)
        self._funcs2 = funcs2
        self._ids2 = ids2
        self._funcs3 = funcs3
        self._ids3 = ids3
        self._keys3 = keys3
        self._keys4 = keys4
        self._T = T
        self._N_zeros_0 = N_zeros_0

        # Pre-allocate working arrays to avoid repeated memory allocation
        self._work_array_1d = np.zeros(N_zeros_0, dtype=np.float64)
        self._work_array_2d = None  # Will be allocated when needed
        self._derivatives_temp = np.zeros(len(funcs3), dtype=np.float64)
        self._secondary_temp = np.zeros(len(funcs2), dtype=np.float64)
        self._initialization_temp = np.zeros(len(funcs1), dtype=np.float64)

        # Assign method references directly for backward compatibility
        self.initialize_by_function = self.initialize_by_function_method
        self.pv_dfdt_global = self.pv_dfdt_update_method
        self.s_u_update = self.s_u_update_method
        self.optimize = self.optimize_method
        self.s_u_residual = self.s_u_residual_method


    def advance_cycle(self, y0, cycleID, step = 1):

        # computes the current time within the cycle
        n_t = self._to.n_c - 1
        end_cycle = cycleID + step
        # retrieves the time points for the current cycle, n_t is the step size
        t = self._to._sym_t.values[cycleID*n_t:end_cycle*n_t+1]

        # solves the system of ODEs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self._method != 'LSODA':
                res = solve_ivp(fun=self.pv_dfdt_global,
                                t_span=(t[0], t[-1]),
                                y0=np.array(y0)[self.perm_indices],
                                t_eval=t,
                                max_step=self.dt,
                                method=self._method,
                                atol=self._atol,
                                rtol=self._rtol,
                                )
            else:
                res = solve_ivp(fun=self.pv_dfdt_global,
                                t_span=(t[0], t[-1]),
                                y0=np.array(y0)[self.perm_indices],
                                t_eval=t,
                                method=self._method,
                                atol=self._atol,
                                rtol=self._rtol,
                                lband=self.lband,
                                uband=self.uband,
                                )

        if res.status == -1:
            return False

        # updates the primary state variables
        y = res.y
        y = y[self.inv_perm_indices]

        # updates the state variables in the DataFrame
        ids = list(self._global_psv_update_fun.keys())
        inds= list(range(len(ids)))
        self._asd.iloc[cycleID*n_t:(end_cycle)*n_t+1, ids] = y[inds, 0:n_t*step+1].T

        if cycleID == 0: return False

        cycleP = end_cycle - 1

        cs   = self._asd[self._cols].iloc[cycleP*n_t:end_cycle*n_t, :].values
        cp   = self._asd[self._cols].iloc[(cycleP-1) *n_t:(cycleP)*n_t, :].values

        cp_ptp = np.max(np.abs(cp), axis=0)
        cp_r   = np.max(np.abs(cs - cp), axis=0)

        test = cp_r / cp_ptp
        test[cp_ptp <= 1e-10] = cp_r[cp_ptp <= 1e-10]
        if np.max(test) > self._step_tol : return False
        return True


    def solve(self):

        # initialize the solution fields
        self._asd.loc[0, self._initialize_by_function.index] = \
            self.initialize_by_function(y=self._asd.loc[0].to_numpy()).T

        # Solve the main system of ODEs..

        for i in range(0, self._to.ncycles, self.step): # step is a pulse, we might wabnt to do it in all pulses
            # print(i)
            y0 = self._asd.iloc[i * (self._to.n_c-1), list(self._global_psv_update_fun.keys())].to_list()
            try:
                # advances the cycle one step at the time, and only that step,
                #changes are to select a range of cycles up to to ith, + dept of cycle instead of selecting that index.
                flag = self.advance_cycle(y0=y0, cycleID=i, step=self.step)
            except ValueError:
                self._Nconv = i-1
                self.converged = False
                break
            if flag and i > self._to.export_min:
                self._Nconv = i + self.step - 1
                self.converged = True
                break
            if i + self.step - 1 == self._to.ncycles - 1:
                self._Nconv = i + self.step - 1
                self.converged = False

        self._to.n_t = (self.Nconv+1)*(self._to.n_c-1) + 1

        self._asd = self._asd.iloc[:self._to.n_t]
        self._to._sym_t   = self._to._sym_t.head(self._to.n_t)
        self._to._cycle_t = self._to._cycle_t.head(self._to.n_t)


        keys4  = np.array(list(self._global_ssv_update_fun.keys()))
        temp   = np.zeros(self._asd.iloc[:,keys4].shape)
        for i, line in enumerate(self._asd.values) :
            line[keys4] = self.s_u_update(0.0, line)
            if self._optimize_secondary_sv:
                temp[i,:] = self.optimize(line, keys4)
            else:
                temp[i,:] = line[keys4]
        self._asd.iloc[:,keys4] = temp

        for key in self._vd.keys():
            self._vd[key]._u = self._asd[key]

    # Class methods for better organization and potential optimization
    def _safe_extract(self, func_result):
        """Safe scalar extraction helper method"""
        return func_result.item() if hasattr(func_result, 'item') and func_result.ndim > 0 else func_result

    def initialize_by_function_method(self, y: np.ndarray[float]) -> np.ndarray[float]:
        """
        Initialize the state variables using a set of initialization functions.
        Class method version for better organization and potential optimization.
        """
        # Use pre-allocated array to avoid memory allocation
        for i, (fun, inds) in enumerate(zip(self._funcs1, self._ids1)):
            self._initialization_temp[i] = self._safe_extract(fun(t=0.0, y=y[inds]))
        
        return self._initialization_temp

    def s_u_update_method(self, t: float, y: np.ndarray[float]) -> np.ndarray[float]:
        """
        Updates the secondary state variables based on the current values of the primary state variables.
        Class method version for better organization and potential optimization.
        """
        # Use pre-allocated array to avoid memory allocation
        for i, (fi, yi) in enumerate(zip(self._funcs2, y[self._ids2])):
            self._secondary_temp[i] = self._safe_extract(fi(t=t, y=yi))
        
        return self._secondary_temp

    def s_u_residual_method(self, y, yall, keys):
        """Function to compute the residual of the secondary state variables."""
        yall[keys] = y
        return (y - self.s_u_update_method(0.0, yall))

    def optimize_method(self, y: np.ndarray, keys):
        """Function to optimize the secondary state variables."""
        yk = y[keys]
        sol = least_squares(
            self.s_u_residual_method,
            yk,
            args=(y, keys),
            ftol=1.0e-5,
            xtol=1.0e-15,
            loss='linear',
            method='lm',
            max_nfev=int(1e6)
        )
        y[keys] = sol.x
        return sol.x

    def pv_dfdt_update_method(self, t: float, y: np.ndarray[float]) -> np.ndarray[float]:
        """
        Function to compute the derivatives of the primary state variables over time.
        Class method version for better organization and potential optimization.
        """
        # calculates the current time within the heart cycle
        ht = t % self._T

        # permutes the primary state variables using index-based operation
        y2 = y[self.inv_perm_indices]

        # Use pre-allocated working arrays to avoid repeated memory allocation
        if len(y.shape) == 2:
            # For 2D arrays, allocate/resize as needed
            if self._work_array_2d is None or self._work_array_2d.shape != (self._N_zeros_0, y.shape[1]):
                self._work_array_2d = np.zeros((self._N_zeros_0, y.shape[1]), dtype=np.float64)
            else:
                self._work_array_2d.fill(0.0)  # Reset instead of allocating
            y_temp = self._work_array_2d
        else:
            # For 1D arrays, use pre-allocated array
            self._work_array_1d.fill(0.0)  # Reset instead of allocating
            y_temp = self._work_array_1d

        # assigns reordered primary state variables to the temporary array
        y_temp[self._keys3] = y2

        # updates the secondary state variables, and optimizes them if necessary
        for _ in range(self._n_sub_iter):
            y_temp[self._keys4] = self.s_u_update_method(t, y_temp)
        if self._optimize_secondary_sv:
            y_temp[self._keys4] = self.optimize_method(y_temp, self._keys4)

        # Compute derivatives using pre-allocated array to avoid memory allocation
        for i, (fi, yi) in enumerate(zip(self._funcs3, y_temp[self._ids3])):
            self._derivatives_temp[i] = self._safe_extract(fi(t=ht, y=yi))
        
        # Apply inverse permutation using index-based operation
        return self._derivatives_temp[self.perm_indices]

    @property
    def vd(self):
        return self._vd


    @property
    def dt(self) -> float:
        return self._to.dt


    @property
    def Nconv(self) -> float:
        return self._Nconv


    @property
    def optimize_secondary_sv(self)->bool:
        return self._optimize_secondary_sv


    @property
    def n_sub_iter(self)->int:
        return self._n_sub_iter


    @n_sub_iter.setter
    def n_sub_iter(self, value):
        assert isinstance(value, int)
        assert value > 0
        self._n_sub_iter = value
