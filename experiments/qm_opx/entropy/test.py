from entropylab import *

def my_func():
    return {'res1':1, 'res2': 2}

node1 = PyNode("node1", my_func,output_vars={'res1'})
node2 = PyNode("node2", my_func,output_vars={'res2'}, input_vars={'in1': node1.outputs['res1']})

db_file='docs_cache/tutorial.db'
db = SqlAlchemyDB(db_file)
experiment_resources = ExperimentResources(db)

experiment = Graph(experiment_resources, {node1, node2}, "run_a") #No resources used here
handle = experiment.run()