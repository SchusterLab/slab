from slab.instruments.SignalCore import SignalCore

#im = InstrumentManager()

sc1 = SignalCore(name="SignalCore", address="100020A0")
sc2 = SignalCore(name="SignalCore", address="100020A1")
sc3 = SignalCore(name="SignalCore", address="1000209F")
sc4 = SignalCore(name="SignalCore", address="100026C2")

#sc1.set_standby(True)
#sc2.set_rf2_standby(True)
#sc3.set_frequency(5e9)
#sc2.set_frequency(12e9)
#sc3.set_frequency(12e9)

#sc1.set_power(10)
#sc2.set_power(30)
sc3.set_power(10)

sc1.close_device()
sc2.close_device()
sc3.close_device()
