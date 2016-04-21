__author__ = 'Nelson'

from slab.experiments.General.SequentialExperiment import *

datapath = os.getcwd() + '\data'

prefix = 'Testing'
data_file =  os.path.join(datapath, get_next_filename(datapath, prefix, suffix='.h5'))


def testing(self):
    print "testing!!!!!!!!!!!!!!!!!" + str(self.expt.offset_freq)


seq_exp = SequentialExperiment()

seq_exp.run(('Ramsey',{'seq_post_run':testing}))
seq_exp.run(('Rabi',{"trigger_period":0.0003,"data_file":data_file}))
seq_exp.run(('Rabi',{"data_file":data_file}))
seq_exp.run(('Rabi',{"data_file":data_file}))
seq_exp.run(('Ramsey',{}))
seq_exp.run(('T1',{}))
