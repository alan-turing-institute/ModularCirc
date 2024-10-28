from .Time import TimeClass
from .StateVariable import StateVariable
from .Models.OdeModel import OdeModel
from .HelperRoutines import bold_text
from pandera.typing import DataFrame, Series
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
        
        self._asd = model.all_sv_data
        self._vd  = model._state_variable_dict
        self._to  = model.time_object
        
        self._global_psv_update_fun  = {}
        self._global_ssv_update_fun  = {}
        self._global_psv_update_fun_n  = {}
        self._global_ssv_update_fun_n  = {}
        self._global_psv_update_ind  = {}
        self._global_ssv_update_ind  = {}
        self._global_psv_names      = []
        self._global_sv_init_fun    = {}
        self._global_sv_init_ind    = {}
        self._global_sv_id          = {key: id   for id, key in enumerate(model.all_sv_data.columns.to_list())}
        self._global_sv_id_rev      = {id: key   for id, key in enumerate(model.all_sv_data.columns.to_list())}
                 
        self._initialize_by_function = pd.Series()
        
        self._N_sv = len(self._global_sv_id)
        
        self._Nconv = None
        
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
        
        for key, component in self._vd.items():
            mkey = self._global_sv_id[key]
            if component.i_func is not None:
                if not suppress_output: print(f" -- Variable {bold_text(key)} added to the init list.")
                if not suppress_output: print(f'    - name of update function: {bold_text(component.i_name)}')
                if not suppress_output: print(f'    - inputs: {component.i_inputs.to_list()}')
                self._initialize_by_function[key] = component
                self._global_sv_init_fun[mkey] = component.i_func
                self._global_sv_init_ind[mkey] = [self._global_sv_id[key2] for key2 in component.i_inputs.to_list()]
                
            if component.dudt_func is not None:
                if not suppress_output: print(f" -- Variable {bold_text(key)} added to the principal variable key list.")
                if not suppress_output: print(f'    - name of update function: {bold_text(component.dudt_name)}')
                if not suppress_output: print(f'    - inputs: {component.inputs.to_list()}')
                self._global_psv_update_fun[mkey]   = component.dudt_func
                self._global_psv_update_fun_n[mkey] = component.dudt_name
                self._global_psv_update_ind[mkey]   = [self._global_sv_id[key2] for key2 in component.inputs.to_list()]
                self._global_psv_update_ind[mkey]   = np.pad(self._global_psv_update_ind[mkey], (0, self._N_sv-len(self._global_psv_update_ind[mkey])), mode='constant', constant_values=-1)
                self._global_psv_names.append(key)
                
            elif component.u_func is not None:
                if not suppress_output: print(f" -- Variable {bold_text(key)} added to the secondary variable key list.")
                if not suppress_output: print(f'    - name of update function: {bold_text(component.u_name)}')
                if not suppress_output: print(f'    - inputs: {component.inputs.to_list()}')
                self._global_ssv_update_fun[mkey]   = component.u_func
                self._global_ssv_update_fun_n[mkey] = component.u_name
                self._global_ssv_update_ind[mkey]   = [self._global_sv_id[key2] for key2 in component.inputs.to_list()]
                self._global_ssv_update_ind[mkey]   = np.pad(self._global_ssv_update_ind[mkey], (0, self._N_sv-len(self._global_ssv_update_ind[mkey])), mode='constant', constant_values=-1)
            else:
                continue
        self._N_psv= len(self._global_psv_update_fun)
        self._N_ssv= len(self._global_ssv_update_fun)
        if not suppress_output: print(' ')
        self.generate_dfdt_functions()
        return
    
    @property
    def vd(self) -> Series[StateVariable]:
        return self._vd
    
    @property
    def dt(self) -> float:
        return self._to.dt
    
    @property
    def Nconv(self) -> float:
        return self._Nconv
    
    @property
    def n_sub_iter(self)->int:
        return self._n_sub_iter
    
    @property
    def optimize_secondary_sv(self)->bool:
        return self._optimize_secondary_sv
    
    @n_sub_iter.setter
    def n_sub_iter(self, value):
        assert isinstance(value, int)
        assert value > 0
        self._n_sub_iter = value
        
    def generate_dfdt_functions(self):
        
        funcs1 = self._global_sv_init_fun.values()
        ids1   = self._global_sv_init_ind.values()

        def initialize_by_function(y:np.ndarray[float]) -> np.ndarray[float]:
            return np.array([fun(t=0.0, y=y[inds]) for fun, inds in zip(funcs1, ids1)])
            
        funcs2 = np.array(list(self._global_ssv_update_fun.values()))
        ids2   = np.stack(list(self._global_ssv_update_ind.values()))
        
        # @nb.njit(cache=True) 
        def s_u_update(t, y:np.ndarray[float,float]) -> np.ndarray[float]:
            return np.array([fi(t=y, y=yi) for fi, yi in zip(funcs2, y[ids2])], dtype=np.float64)
        
        def s_u_residual(y, yall, keys):
            yall[keys] = y
            return (y - s_u_update(0.0, yall))
        
        def optimize(y:np.ndarray, keys):
            yk = y[keys]
            sol = least_squares(   # root
                s_u_residual, 
                yk, 
                args=(y, keys), 
                ftol=1.0e-5,
                xtol=1.0e-15,
                loss='linear',
                method='lm',
                max_nfev=int(1e6)
                )
            y[keys] = sol.x
            return sol.x  # sol.x
        
        keys3  = np.array(list(self._global_psv_update_fun.keys()))
        keys4  = np.array(list(self._global_ssv_update_fun.keys()))
        funcs3 = np.array(list(self._global_psv_update_fun.values()))
        ids3   = np.stack(list(self._global_psv_update_ind.values()))
        
        T = self._to.tcycle
        N_zeros_0 = len(self._global_sv_id)
        _n_sub_iter = self._n_sub_iter
        _optimize_secondary_sv = self._optimize_secondary_sv
        
        keys3_dict = dict()
        for key, line in zip(keys3,ids3):
            line2= [val for val in np.unique(line) if val != -1]
            keys3_dict[key] = set(line2)
        
        keys3_back_dict = dict()
        for key, val in enumerate(keys3_dict):
            keys3_back_dict[val] = key
            
        keys4_dict = dict()
        for key, line in zip(keys4,ids2):
            line2= [val for val in np.unique(line) if val != -1]
            keys4_dict[key] = set(line2)
        
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
        
        mat = np.zeros((len(sparsity_map),len(sparsity_map)))
        for key, rows in sparsity_map.items():
            mat[key, np.array(list(rows), dtype=np.int64)] = 1
        
        sparse_mat = csr_matrix(mat)
        perm = reverse_cuthill_mckee(sparse_mat, symmetric_mode=False)
        perm_mat = np.zeros((len(perm), len(perm)))
        for i,j in enumerate(perm):
            perm_mat[i,j] = 1
            
        self.perm_mat = perm_mat
        
        sparse_mat_reordered = sparse_mat[perm, :][:, perm]

        sparse_mat_reordered_indexes = np.argwhere(sparse_mat_reordered.toarray())
        temp = sparse_mat_reordered_indexes[:,0] - sparse_mat_reordered_indexes[:,1]
        uband = np.abs(np.min(temp))
        lband = np.max(temp)

        self.lband = lband
        self.uband = uband
        
        def pv_dfdt_update(t, y:np.ndarray[float]) -> np.ndarray[float]:
            ht = t%T
            y2 = perm_mat.T @ y
            if len(y.shape) == 2:
                y_temp = np.zeros((N_zeros_0,y.shape[1]))
            else:
                y_temp = np.zeros((N_zeros_0))
            y_temp[keys3] = y2
            for _ in range(_n_sub_iter):
                y_temp[keys4] = s_u_update(t, y_temp)  
            if _optimize_secondary_sv:
                y_temp[keys4] = optimize(y_temp, keys4)
            return perm_mat @ np.fromiter([fi(t=ht, y=yi) for fi, yi in zip(funcs3, y_temp[ids3])], dtype=np.float64)
        
        def pv_jac_packed(t, y:np.ndarray[float]) -> np.ndarray[float]:
            ht = t%T
            funcs3_reordered = funcs3[perm]
            jac_packed = np.zeros((2*lband+uband+1,len(keys3)))
            eps = 1e-3
            y2 = y.copy()
            
            for i, j in sparse_mat_reordered_indexes:
                y2[j] += eps
                f_plus = funcs3_reordered[i](t=ht, y=y2)
                y2[j] -= 2*eps
                f_min  = funcs3_reordered[i](t=ht, y=y2)
                y2[j] += eps
                jac_packed[uband+i-j,j] = (f_plus - f_min) / 2. / eps
            
            return jac_packed
            
        
        self.initialize_by_function = initialize_by_function    
        self.pv_dfdt_global = pv_dfdt_update
        self.s_u_update     = s_u_update
        
        self.pv_jac_packed = pv_jac_packed
        
        self.optimize = optimize
        self.s_u_residual = s_u_residual
    
    def advance_cycle(self, y0, cycleID):
        n_t = self._to.n_c - 1
        t = self._to._sym_t.values[cycleID*n_t:(cycleID+1)*n_t+1] 
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self._method != 'LSODA':
                res = solve_ivp(fun=self.pv_dfdt_global, 
                                t_span=(t[0], t[-1]), 
                                y0=self.perm_mat @ y0, 
                                t_eval=t,
                                max_step=self._to.dt,
                                method=self._method,
                                atol=self._atol,
                                rtol=self._rtol,
                                )
            else:
                res = solve_ivp(fun=self.pv_dfdt_global, 
                                t_span=(t[0], t[-1]), 
                                y0=self.perm_mat @ y0, 
                                t_eval=t,
                                max_step=self._to.dt,
                                method=self._method,
                                atol=self._atol,
                                rtol=self._rtol,
                                lband=self.lband,
                                uband=self.uband,
                                # jac=self.pv_jac_packed
                                )

        if res.status == -1:
            return False
        y = res.y
        y = self.perm_mat.T @ y
        for ind, id in enumerate(self._global_psv_update_fun.keys()):
            self._asd.iloc[cycleID*n_t:(cycleID+1)*n_t+1, id] = y[ind, 0:n_t+1]
            
        if cycleID == 0: return False
        
        ind = list(self._global_psv_update_fun.keys())
        cycleP = cycleID - 1
        if self._conv_cols is None:
            cols = [col for col in self._asd.columns if 'v_' in col or 'p_' in col]
        else:
            cols = self._conv_cols
        cs   = self._asd[cols].iloc[cycleID*n_t:(cycleID+1)*n_t, :].values
        cp   = self._asd[cols].iloc[cycleP *n_t:(cycleP +1)*n_t, :].values
        
        cp_ptp = np.max(np.abs(cp), axis=0)
        cp_r   = np.max(np.abs(cs - cp), axis=0)
        
        for val, nval in zip(cp_r, cp_ptp):
            if nval > 1e-10:
                if val / nval > self._step_tol: return False
            else:
                if val > self._step_tol: return False
        return True
    
    
    def solve(self):
        # initialize the solution fields
        self._asd.loc[0, self._initialize_by_function.index] = \
            self.initialize_by_function(y=self._asd.loc[0].to_numpy()).T
        # Solve the main system of ODEs.. 
        for i in range(self._to.ncycles):
            # print(i)
            y0 = self._asd.iloc[i * (self._to.n_c-1), list(self._global_psv_update_fun.keys())].to_list()
            try:
                flag = self.advance_cycle(y0=y0, cycleID=i)
            except ValueError:
                self._Nconv = i-1
                self.converged = False
                break
            if flag and i > self._to.export_min:
                self._Nconv = i
                self.converged = True
                break
            if i == self._to.ncycles - 1:
                self._Nconv = i
                self.converged = False
        
        self._asd = self._asd.iloc[:self.Nconv*(self._to.n_c)+1]
        self._to._sym_t   = self._to._sym_t.head(self.Nconv*(self._to.n_c)+1)
        self._to._cycle_t = self._to._cycle_t.head(self.Nconv*(self._to.n_c)+1)
        
        self._to.n_t = self.Nconv*(self._to.n_c-1)+1
            
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
        

            
        
        
        
        
    
    
    
    
    