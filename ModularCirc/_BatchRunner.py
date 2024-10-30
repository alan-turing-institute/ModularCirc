import numpy as np
import pandas as pd
import json
import joblib
import os

from tqdm import tqdm
from scipy.stats.qmc import (LatinHypercube, Sobol, Halton, QMCEngine, scale)

from .Models.OdeModel import OdeModel
from .Models.ParametersObject import ParametersObject
from .Solver import Solver

DEFAULT_RANDOM_SEED = 42

sampler_dictionary = {
    'LHS' : LatinHypercube,
    'Sobol' : Sobol,
    'Halton': Halton,
}

class _BatchRunner:
    def __init__(self, sampler:str='LHS', seed=DEFAULT_RANDOM_SEED) -> None:
        self._sample_generator = sampler_dictionary[sampler]
        self._seed      = seed
        return
        
    def setup_sampler(self, template_json):
        with open(template_json) as file:
            self._template : dict = json.load(file)
        self._parameters_2_sample = dict()
        self._parameters_constant = dict()
        
        self.condense_dict_parameters(self._template)
        self._d = len(self._parameters_2_sample)
        self._l_bounds = [bounds[0] for bounds in self._parameters_2_sample.values()]
        self._u_bounds = [bounds[1] for bounds in self._parameters_2_sample.values()]
        
        self._sampler : QMCEngine = self._sample_generator(d=self._d, seed=self._seed)    
        return
        
    def condense_dict_parameters(self, dict_param, prev=''):
        for key, val in dict_param.items():
            if len(prev) > 0:
                new_key = prev.split('.')[-1] + '.' + key
            else:
                new_key = key
            if isinstance(val, dict):
                self.condense_dict_parameters(val, new_key)
            else:
                if len(val) > 1:
                    value, r = val
                    self._parameters_2_sample[new_key] = tuple(np.array(r) * value)
                else:
                    self._parameters_constant[new_key] = val[0]
        return  
    
    def sample(self, nsamples:int):
        samples  = self._sampler.random(nsamples)
        samples_scaled = scale(sample=samples, l_bounds=self._l_bounds, u_bounds=self._u_bounds)
        self._samples  = pd.DataFrame(samples_scaled, columns=list(self._parameters_2_sample.keys()))
        for key, val in self._parameters_constant.items():
            self._samples[key] = val
        return
            
    def map_sample_timings(self, map:dict = dict(), ref_time=1.0):
        for key, mappings in map.items():
            for key2 in mappings:
                if key2 in self._samples.columns: self._samples.drop(key2, axis=1)
                self._samples[key2] = self._samples[key] * self._samples['T'] / ref_time
            if key not in mappings : self._samples.drop(key, inplace=True)
        self._ref_time = ref_time
        return
    
    def map_vessel_volume(self):
        vessels = list(self._template['VESSELS'].keys())
        vessels_c_names =[vessel + '.c' for vessel in vessels]
        
        samples_vessels_c = self._samples[vessels_c_names]
        tot_vessels_c     = samples_vessels_c.sum(axis=1)
        
        for vessel in vessels:
            self._samples[vessel + '.v'] = self._samples['v_tot'] * self._samples[vessel + '.c'] / tot_vessels_c
        self._samples.drop('v_tot', axis=1, inplace=True)
        return
        
        
    def setup_model(self, model:OdeModel, po:ParametersObject, time_setup:dict):
        self._model_generator = model
        self._po_generator    = po
        self._tst             = time_setup
        return
    
        
    def run_batch(self, n_jobs=1, **kwargs):
        if n_jobs == 1:
            success = self._samples.apply(lambda row : self._run_case(row, **kwargs), axis=0)
        else:
            success = joblib.Parallel(n_jobs=n_jobs)(
                                        joblib.delayed(self._run_case)(row, **kwargs)
                                        for _, row in tqdm(self._samples.iterrows(), total=len(self._samples))
                                     )
        return success
    
    def _run_case(self, row, **kwargs):
        time_setup = self._tst.copy()
        time_setup['T']  = row['T']
        time_setup['dt']*= row['T'] / self._ref_time
        
        po : ParametersObject = self._po_generator()
        for key, val in row.items():
            if key == "T":
                continue
            try:
                obj, param = key.split(".")
            except:
                raise Exception(key)
                
            po._set_comp(obj,[obj,], **{param: val})

        model : OdeModel = self._model_generator(time_setup_dict = time_setup, parobj=po, suppress_printing=True)
        
        solver = Solver(model=model)
        
        if 'optimize_secondary_sv' in kwargs:
            optimize_secondary_sv = kwargs['optimize_secondary_sv']
        else:
            optimize_secondary_sv = False
            
        if 'conv_cols' in kwargs:
            conv_cols = kwargs['conv_cols']
        else:
            conv_cols = None
            
        if 'method' in kwargs:
            method = kwargs['method']
        else:
            method = 'LSODA'
        
        if 'out_cols' in kwargs:
            out_cols = kwargs['out_cols']
        else:
            out_cols = None
            
        if 'output_path' in kwargs:
            output_path = kwargs['output_path']
        else:
            output_path = None

        solver.setup(
            suppress_output=True,
            optimize_secondary_sv=optimize_secondary_sv,
            conv_cols=conv_cols,
            method=method
        )    
        solver.solve()
        
        if not solver.converged: return False
        
        if out_cols is None:
            raw_signal = solver._asd.copy()
        else:
            raw_signal = solver._asd[out_cols].copy()

            
        if output_path is not None: raw_signal.to_csv(os.path.join(output_path, f'all_outputs_{row.name}.csv'))
        
        raw_signal_short = raw_signal.tail(model.time_object.n_c)
        raw_signal_short.index = pd.MultiIndex.from_tuples([(row.name, i) for i in range(len(raw_signal_short))], names=
        ['realization', 'time_ind'])
        
        # raw_signal_short.reindex([ i for i in range(len(raw_signal_short))])
        # raw_signal_short.index.name = 'Index (realization, time)'
        return raw_signal_short
        
            
        
        
