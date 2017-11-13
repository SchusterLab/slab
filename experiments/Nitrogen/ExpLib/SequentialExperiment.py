__author__ = 'Nelson'

from slab.experiments.General.run_experiment import *
from slab.experiments.Multimode.run_multimode_experiment import *
import numbers
from slab import *
import json
import collections

import gc


def update_dict(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            r = update_dict(d.get(k, {}), v)
            d[k] = r
        else:
            d[k] = u[k]
    return d


class SequentialExperiment():
    def __init__(self, lp_enable=True):
        self.expt = None
        self.lp_enable = lp_enable
        # config_file = 'config.json'
        # datapath = os.getcwd()

    def run(self, expt_name, vary_dict={}, expt_kwargs={}):

        if self.expt is not None:
            del self.expt
            gc.collect()

        datapath = os.getcwd() + '\data'
        config_file = os.path.join(datapath, "..\\config" + ".json")
        with open(config_file, 'r') as fid:
            cfg_str = fid.read()
        cfg_dict = json.loads(cfg_str)

        cfg_dict_temp = update_dict(cfg_dict, vary_dict)
        with open('config.json', 'w') as fp:
            json.dump(cfg_dict_temp, fp)

        print("--------Parameters varied in config file ---------")
        print(vary_dict)
        print("--------------------------------------------------")
        print("--------Parameters changed through extra_args---------")
        print(expt_kwargs)
        print("--------------------------------------------------")

        ## automatically save kwargs to data_file
        if 'data_file' in expt_kwargs:
            data_file = expt_kwargs['data_file']
            for key in expt_kwargs:
                if isinstance(expt_kwargs[key], numbers.Number) and not key == 'update_config':
                    with SlabFile(data_file) as f:
                        f.append_pt(key, float(expt_kwargs[key]))
                        f.close()

        if 'seq_pre_run' in expt_kwargs:
            expt_kwargs['seq_pre_run'](self)

        self.expt = run_experiment(expt_name, self.lp_enable, config_file='..\\config.json', **expt_kwargs)
        if self.expt is None:
            self.expt = run_multimode_experiment(expt_name, self.lp_enable, **expt_kwargs)

        if 'seq_post_run' in expt_kwargs:
            expt_kwargs['seq_post_run'](self)

        if 'update_config' in expt_kwargs:
            if expt_kwargs['update_config']:
                self.save_config()

    def save_config(self):
        self.expt.save_config()
        print("config saved!")





