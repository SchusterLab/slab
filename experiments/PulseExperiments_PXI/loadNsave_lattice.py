import json
    
def load_lattice_to_quantum_device(lattice_cfg_name, quantum_device_cfg_name, qb_id, setup_id):

    with open(quantum_device_cfg_name, 'r') as f:
        quantum_device_cfg = json.load(f)
    with open(lattice_cfg_name, 'r') as f:
        lattice_cfg = json.load(f)

    for category in quantum_device_cfg.keys():
        #check if category has "A" and "B" entries
        if isinstance(quantum_device_cfg[category], dict) and setup_id in quantum_device_cfg[category].keys():

            #if "A" and "B" are dictionaries where have to walk through keys
            if isinstance(quantum_device_cfg[category][setup_id], dict):
                for key in quantum_device_cfg[category][setup_id]:
                    try:
                        quantum_device_cfg[category][setup_id][key] = lattice_cfg[category][key][qb_id]
                    except:
                        print("[{}][{}] does not exist as a category and key that have as value a qubit list in "
                              "lattice config".format(category, key))

            #else just paste lattice into quantum_device [category][setup_id] directly
            else:
                try:
                    quantum_device_cfg[category][setup_id] = lattice_cfg[category][qb_id]
                except:
                    print("[{}] does not exist as a category that has as value a qubit listin lattice config".format(
                        category))

        #if category isn't a dictionary with setup_id, but has a matching list in lattice_cfg, fill it in
        else:
            if category in lattice_cfg.keys() and len(lattice_cfg[category])==8:
                quantum_device_cfg[category] = lattice_cfg[category][qb_id]



    with open(quantum_device_cfg_name, 'w') as f:
        json.dump(quantum_device_cfg, f, indent=2)

                

def load_quantum_device_to_lattice(lattice_cfg_name, quantum_device_cfg_name, qb_id, setup_id):
    with open(quantum_device_cfg_name, 'r') as f:
        quantum_device_cfg = json.load(f)
    with open(lattice_cfg_name, 'r') as f:
        lattice_cfg = json.load(f)

    for category in lattice_cfg.keys():

        #if category is directly a list of 8 qubit values, find the corresponsing entry in quantum device cfg and
        # stuff it in there
        if isinstance(lattice_cfg[category], list) and len(lattice_cfg[category])== 8:

            #check if category even exists in quantum_device_config
            if category in quantum_device_cfg.keys():

                #check if needs to be stuffed into a setupid dict or just stuffed directly
                if isinstance(quantum_device_cfg[category], dict) and setup_id in quantum_device_cfg[category].keys():
                    lattice_cfg[category][qb_id] = quantum_device_cfg[category][setup_id]
                else:
                    lattice_cfg[category][qb_id] = quantum_device_cfg[category]

            else:
                print("[{}] not a category quantum device config".format(category))

        #if category is a dictionary, walk through the keys. if one of them is a list of eight qubit values,
        # stuff it in quantum device config
        elif isinstance(lattice_cfg[category], dict):
            for key in lattice_cfg[category]:
                if isinstance(lattice_cfg[category][key], list) and len(lattice_cfg[category][key])==8:

                    try:
                        if isinstance(quantum_device_cfg[category], dict) and setup_id in quantum_device_cfg[
                            category].keys():
                            lattice_cfg[category][key][qb_id] = quantum_device_cfg[category][setup_id][key]
                        else:
                            lattice_cfg[category][key][qb_id] = quantum_device_cfg[category][key]
                    except:
                        print("[{}][{}] not a category and key quantum device config".format(category, key))


    with open(lattice_cfg_name, 'w') as f:
        json.dump(lattice_cfg, f, indent=2)


lattice_cfg_name = 'test_sawtooth_lattice_device_config.json'
quantum_device_cfg_name=  'test_quantum_device_config.json'

load_lattice_to_quantum_device(lattice_cfg_name, quantum_device_cfg_name, qb_id = 5, setup_id='2')
#load_quantum_device_to_lattice(lattice_cfg_name, quantum_device_cfg_name, qb_id=0, setup_id = '1')