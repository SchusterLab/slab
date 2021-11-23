from slab.instruments import InstrumentManager

im = InstrumentManager()

pnax = im['PNAX2']
print (pnax.get_id())

# note: switch to cw and then back to pulsed, all pulse parameters remain the same
# still need to restart trigger manually

is_pulsed = 1

if is_pulsed:

    # turning on the pulses
    pnax.write("SENS:PULS0 1")  # automatically sync ADC to pulse gen
    pnax.write("SENS:PULS1 1")
    pnax.write("SENS:PULS2 1")
    pnax.write("SENS:PULS3 1")
    pnax.write("SENS:PULS4 1")
    # turning off the inverting
    pnax.write("SENS:PULS1:INV 0")
    pnax.write("SENS:PULS2:INV 0")
    pnax.write("SENS:PULS3:INV 0")
    pnax.write("SENS:PULS4:INV 0")

else:
    print ("turning off the pulses")
    #
    # pnax.write("SENS:PULS0 1")  # automatically sync ADC to pulse gen
    # pnax.write("SENS:PULS1 1")
    # pnax.write("SENS:PULS2 1")
    # pnax.write("SENS:PULS3 1")
    # pnax.write("SENS:PULS4 1")
    # turning on the inverting
    pnax.write("SENS:PULS1:INV 1")
    pnax.write("SENS:PULS2:INV 1")
    pnax.write("SENS:PULS3:INV 1")
    pnax.write("SENS:PULS4:INV 1")
