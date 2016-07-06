from M8195A import *

m8195A = M8195A(address ='192.168.14.234:5025')

print "Testing set enabled:"
m8195A.set_enabled(1,True)
m8195A.set_enabled(2,False)
print m8195A.get_enabled(1)
print m8195A.get_enabled(2)

print "Testing set amplitude:"
m8195A.set_amplitude(1,0.2)
print m8195A.get_amplitude(1)

print "Testing set analogHigh:"
m8195A.set_analog_high(3,0.5)
print m8195A.get_analog_high(3)

print "Testing set analoLow:"
m8195A.set_analog_low(3,-0.5)
print m8195A.get_analog_low(3)

print "Testing set offset:"
m8195A.set_offset(1,0.05)
print m8195A.get_offset(1)

print m8195A.get_id()

print m8195A.get_dac_mode()

m8195A.set_dac_sample_rate_divider(4)
print m8195A.get_sample_clock_divider()

print "Done!"