from .Models.OdeModel import OdeModel
from .HelperRoutines import bold_text
from .Models.OdeModel import OdeModel

import pandas as pd
import numpy as np
import numba as nb

from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee

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

        # Number of primary and secondary state variables (initialized in setup)
        self._N_psv = 0  # Number of primary state variables
        self._N_ssv = 0  # Number of secondary state variables

        # Variable to store the number of converged cycles.
        self._Nconv = None

        # Number of sub-iterations for the solver.
        self._n_sub_iter = 1

        # flag for checking if the model is converged or not...
        self.converged = False

    def _pad_index_array(self, index_array):
        """
        Helper method to pad index arrays to the length of the state variable array.
        This eliminates code duplication between primary and secondary variable processing.
        """
        return np.pad(index_array,
                     (0, self._N_sv - len(index_array)),
                     mode='constant', constant_values=-1)

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
                self._global_psv_update_ind[mkey]   = self._pad_index_array(self._global_psv_update_ind[mkey])

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
                self._global_ssv_update_ind[mkey]   = self._pad_index_array(self._global_ssv_update_ind[mkey])
            else:
                continue

        if not suppress_output: print(' ')

        # Update counts of primary and secondary state variables
        self._N_psv = len(self._global_psv_update_fun)
        self._N_ssv = len(self._global_ssv_update_fun)
        self._N_sv = len(self._global_sv_id)
        
        if not suppress_output: 
            print(f" -- Total primary state variables: {bold_text(str(self._N_psv))}")
            print(f" -- Total secondary state variables: {bold_text(str(self._N_ssv))}")
            print(' ')

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

        # Pre-compute frequently used key arrays to avoid repeated computation
        self._cached_keys4 = keys4  # Already computed above
        self._cached_psv_keys = list(self._global_psv_update_fun.keys())  # Primary state variable keys



        # Pre-allocate working arrays to avoid repeated memory allocation
        self._work_array_1d = np.zeros(self._N_sv, dtype=np.float64)
        self._derivatives_temp = np.zeros(self.N_psv, dtype=np.float64)
        self._secondary_temp = np.zeros(self.N_ssv, dtype=np.float64)
        self._initialization_temp = np.zeros(len(funcs1), dtype=np.float64)

        # Pre-compute function-index pairs for hot path optimization
        # Since _funcs3 and _ids3 never change, compute the pairs once
        self._func_index_pairs3 = list(zip(self._funcs3, self._ids3))
        self._func_index_pairs2 = list(zip(self._funcs2, range(len(self._funcs2))))
        self._func_index_pairs1 = list(zip(self._funcs1, range(len(self._funcs1))))

        # Pre-compute constants for advance_cycle optimization
        self._n_t_minus_1 = self._to.n_c - 1  # Time points per cycle minus 1
        self._primary_indices = np.arange(len(self._cached_psv_keys))  # Avoid list(range()) 
        
        # Pre-allocate arrays for advance_cycle to avoid repeated allocations
        self._y0_permuted = np.zeros(len(self._cached_psv_keys), dtype=np.float64)
        self._convergence_tolerance = 1e-10  # Cache tolerance value

        # Assign method references directly for backward compatibility
        self.initialize_by_function = self.initialize_by_function_method
        self.pv_dfdt_global = self.pv_dfdt_update_method
        self.s_u_update = self.s_u_update_method
        self.optimize = self.optimize_method
        self.s_u_residual = self.s_u_residual_method

        N_psv = self._N_psv
        temp_func3 = tuple(self._funcs3)
        ids3 = self._ids3  # Capture ids3 in local scope

        # for func in self._funcs3:
        #     print(func.__name__)
        # raise Exception
        # @nb.njit('float64[:](float64, float64[:])', cache=True)
        # def compute_pv_dfdt_func_iteration(ht, y):
        #     return [temp_func3[i](ht, y[ids3[i]]) for i in range(N_psv)]
        # self.compute_pv_dfdt_func_iteration = compute_pv_dfdt_func_iteration

    def advance_cycle(self, y0, cycleID, step = 1):
        """
        Optimized advance_cycle method with reduced allocations and computations.
        """
        # Use pre-computed constants to avoid repeated calculations
        n_t = self._n_t_minus_1
        end_cycle = cycleID + step
        
        # More efficient time slice extraction (single slice operation)
        start_idx = cycleID * n_t
        end_idx = end_cycle * n_t + 1
        t = self._to._sym_t.values[start_idx:end_idx]

        # Optimize initial condition preparation - reuse pre-allocated array
        np.copyto(self._y0_permuted, y0)  # Copy to pre-allocated array
        y0_permuted = self._y0_permuted[self.perm_indices]  # Then permute

        # solves the system of ODEs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self._method != 'LSODA':
                res = solve_ivp(fun=self.pv_dfdt_global,
                                t_span=(t[0], t[-1]),
                                y0=y0_permuted,
                                t_eval=t,
                                max_step=self.dt,
                                method=self._method,
                                atol=self._atol,
                                rtol=self._rtol,
                                )
            else:
                res = solve_ivp(fun=self.pv_dfdt_global,
                                t_span=(t[0], t[-1]),
                                y0=y0_permuted,
                                t_eval=t,
                                method=self._method,
                                atol=self._atol,
                                rtol=self._rtol,
                                lband=self.lband,
                                uband=self.uband,
                                )

        if res.status == -1:
            return False

        # Optimize state variable updates - reduce array operations
        y = res.y[self.inv_perm_indices]  # Combine permutation with result extraction

        # Use pre-computed indices to avoid list creation
        ids = self._cached_psv_keys
        n_time_steps = n_t * step + 1
        self._asd.iloc[start_idx:end_idx, ids] = y[self._primary_indices, :n_time_steps].T

        # Early return for first cycle
        if cycleID == 0: 
            return False

        # Optimized convergence check with reduced DataFrame operations
        cycleP = end_cycle - 1
        
        # Single iloc call per DataFrame section (more efficient)
        current_start = cycleP * n_t
        current_end = end_cycle * n_t
        previous_start = (cycleP - 1) * n_t
        previous_end = cycleP * n_t
        
        # Extract convergence data in one operation each
        cs = self._asd[self._cols].iloc[current_start:current_end, :].values
        cp = self._asd[self._cols].iloc[previous_start:previous_end, :].values

        # Vectorized convergence test with cached tolerance
        cp_ptp = np.max(np.abs(cp), axis=0)
        cp_r = np.max(np.abs(cs - cp), axis=0)

        # Optimized convergence calculation using pre-cached tolerance
        test = np.divide(cp_r, cp_ptp, out=cp_r.copy(), where=(cp_ptp > self._convergence_tolerance))
        test[cp_ptp <= self._convergence_tolerance] = cp_r[cp_ptp <= self._convergence_tolerance]
        
        return np.max(test) <= self._step_tol


    def solve(self):

        # initialize the solution fields
        self._asd.loc[0, self._initialize_by_function.index] = \
            self.initialize_by_function(y=self._asd.loc[0].to_numpy()).T

        # Solve the main system of ODEs..

        for i in range(0, self._to.ncycles, self.step): # step is a pulse, we might wabnt to do it in all pulses
            # print(i)
            y0 = self._asd.iloc[i * (self._to.n_c-1), self._cached_psv_keys].to_list()
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


        # Use pre-computed keys4 array to avoid recomputation
        keys4 = self._cached_keys4
        
        # Vectorized batch processing for secondary state variables
        data_array = self._asd.values  # Convert DataFrame to numpy array for faster access
        secondary_results = self.process_secondary_variables_batch(data_array, keys4, batch_size=1000)
        
        # Update the DataFrame with processed results
        self._asd.iloc[:,keys4] = secondary_results

        for key in self._vd.keys():
            self._vd[key]._u = self._asd[key]

    # Class methods for better organization and potential optimization
    def _safe_extract(self, func_result):
        """Safe scalar extraction helper method"""
        return func_result.item() if hasattr(func_result, 'item') and func_result.ndim > 0 else func_result
    
    def _compute_derivatives_optimized(self, ht: float, y_temp: np.ndarray):
        """
        Optimized derivative computation that minimizes Python overhead.
        Uses vectorized input extraction - the fastest approach tested.
        """
        all_inputs = y_temp[self._ids3]  # NumPy's optimized vectorized indexing
        
        results = self._derivatives_temp
        funcs = self._funcs3
        
        for i in range(self.N_psv):
            func_result = funcs[i](t=ht, y=all_inputs[i])
            results[i] = func_result

    def initialize_by_function_method(self, y: np.ndarray[float]) -> np.ndarray[float]:
        """
        Initialize the state variables using a set of initialization functions.
        Vectorized version for better performance.
        """
        # Use pre-computed function-index pairs for consistent optimization
        results = [fi(t=0.0, y=y[self._ids1[i]]) for fi, i in self._func_index_pairs1]
        
        # Copy results to pre-allocated array
        self._initialization_temp[:] = results
        
        return self._initialization_temp

    def s_u_update_method(self, t: float, y: np.ndarray[float]) -> np.ndarray[float]:
        """
        Updates the secondary state variables based on the current values of the primary state variables.
        Vectorized version for better performance.
        """
        # Create input arrays in one vectorized operation
        y_inputs = y[self._ids2]
        
        funcs = self._funcs2
        results = self._secondary_temp
        
        for i in range(self.N_ssv):
            results[i] = funcs[i](t=t, y=y_inputs[i])
        
        return self._secondary_temp

    def s_u_update_batch_method(self, t: float, y_batch: np.ndarray[float]) -> np.ndarray[float]:
        """
        Batch version of s_u_update_method that processes multiple rows simultaneously.
        
        Args:
            t: Time parameter
            y_batch: 2D array where each row is a state vector (shape: [n_rows, n_state_vars])
            
        Returns:
            2D array of secondary state variable updates (shape: [n_rows, n_secondary_vars])
        """
        n_rows = y_batch.shape[0]
        n_secondary = len(self._funcs2)
        
        # Pre-allocate result array
        results_batch = np.zeros((n_rows, n_secondary), dtype=np.float64)
        
        # Process each secondary function across all rows
        for func_idx, (fi, _) in enumerate(self._func_index_pairs2):
            # Extract input indices for this function
            input_indices = self._ids2[func_idx]
            
            # Get inputs for all rows for this function (vectorized slicing)
            y_inputs_batch = y_batch[:, input_indices]
            
            # Apply function to each row (still need individual calls due to function signature)
            for row_idx in range(n_rows):
                result = fi(t=t, y=y_inputs_batch[row_idx])
                results_batch[row_idx, func_idx] = result
        
        return results_batch

    def process_secondary_variables_batch(self, data_batch: np.ndarray[float], keys4: np.ndarray, batch_size: int = 1000) -> np.ndarray[float]:
        """
        Process secondary state variables in batches for improved performance.
        
        Args:
            data_batch: 2D array of state variable data (shape: [n_rows, n_state_vars])
            keys4: Array of secondary state variable column indices
            batch_size: Number of rows to process simultaneously
            
        Returns:
            2D array of processed secondary state variables (shape: [n_rows, n_secondary_vars])
        """
        n_rows = data_batch.shape[0]
        n_secondary = len(keys4)
        result = np.zeros((n_rows, n_secondary), dtype=np.float64)
        
        # Process data in batches to manage memory usage
        for start_idx in range(0, n_rows, batch_size):
            end_idx = min(start_idx + batch_size, n_rows)
            batch = data_batch[start_idx:end_idx].copy()  # Work on a copy to avoid side effects
            
            # Update secondary variables for this batch
            secondary_updates = self.s_u_update_batch_method(t=0.0, y_batch=batch)
            
            # Apply updates back to batch data
            batch[:, keys4] = secondary_updates
            
            if self._optimize_secondary_sv:
                # For optimization, we still need row-by-row processing due to least_squares API
                for i, row in enumerate(batch):
                    result[start_idx + i, :] = self.optimize_method(row, keys4)
            else:
                # Direct assignment for non-optimized case
                result[start_idx:end_idx, :] = secondary_updates
        
        return result

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
        self._work_array_1d.fill(0.0)  # Reset instead of allocating
        y_temp = self._work_array_1d

        # assigns reordered primary state variables to the temporary array
        y_temp[self._keys3] = y2

        # updates the secondary state variables, and optimizes them if necessary
        for _ in range(self._n_sub_iter):
            y_temp[self._keys4] = self.s_u_update_method(t, y_temp)
        if self._optimize_secondary_sv:
            y_temp[self._keys4] = self.optimize_method(y_temp, self._keys4)

        # Smart vectorized approach: use bulk operations where possible
        # Since the functions are heterogeneous but have similar computational patterns,
        # we can optimize by reducing Python overhead and leveraging NumPy operations
        
        # Method: Pre-extract all input slices and use optimized batch calling
        self._compute_derivatives_optimized(ht, y_temp)
        # self._derivatives_temp = self.compute_pv_dfdt_func_iteration(ht, y_temp)
        
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

    @property
    def N_psv(self) -> int:
        """Number of primary state variables."""
        return self._N_psv

    @property
    def N_ssv(self) -> int:
        """Number of secondary state variables."""
        return self._N_ssv

    @property
    def N_sv(self) -> int:
        """Total number of state variables."""
        return self._N_sv

    @n_sub_iter.setter
    def n_sub_iter(self, value):
        assert isinstance(value, int)
        assert value > 0
        self._n_sub_iter = value
