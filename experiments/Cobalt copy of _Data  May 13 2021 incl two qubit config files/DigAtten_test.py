from slab import InstrumentManager
import json

with open('quantum_device_config.json', 'r') as f:
    quantum_device_cfg  = json.load(f)
with open('experiment_config.json', 'r') as f:
    experiment_cfg = json.load(f)
with open('hardware_config.json', 'r') as f:
    hardware_cfg = json.load(f)

im = InstrumentManager()

# Init digital attenuators
attens = [im[atten] for atten in hardware_cfg['drive_attens']]

# Get attenuator ID
attens[1].get_id()

# Set attenuator to a specific value
attens[1].set_attenuator(-10.0)

# Get attenuator value
attens[1].get_attenuator()
