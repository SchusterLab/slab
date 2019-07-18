# Slab PXI - how to run

# Overall Structure
There are five separate kinds of files:
- instrument manager: you run a script before anything else, connects instruments to python code (need to do this first)
- config files: 3 JSON files (experiment_config, hardware_config, quantum_device_config) where you input all the information you need to configure hardware and your experiment (from which ports you are using on your device, to how many input drives you expect to have, to expected T1 and frequency times of your qubits)
- Pulse generation files: input config files and your experiment name -> you get out a Pulse Sequence object
- Experiment files: input experiment name, config file, and Pulse Sequence object -> it configures your hardware instruments, loads up pulse sequence, and gets everything ready to run. it returns an experiment object
- run_experiment.py -> a short script that calls the pulse sequence files and experiment files; where you input stuff and actual run the command (Experiment Object).run()

# How to Install the Python stuff you need

# Running an experiment
First, you have to do some terminal stuff!
1. Open a terminal window. Run 

#More in depth on each kind of file
## Instrument Manager

## run_experiment.py
Imports:
- Pulse Sequences Class
- Experiment class
- json

sets `path = os.getcwd()`

has a list of experiments that have been written/that you can reasonably call

# Todo:
create a data folder automatically if one doesn't exist already
