__author__ = 'Nelson Leung'

import unittest
from .M8195A import *

class M8195ATest(unittest.TestCase):
    def test(self):
        print("Testing connection")
        m8195A = M8195A(address ='192.168.14.234:5025')

        self.assertEqual(m8195A.get_id()[:8],'Keysight')

        print("Testing set enabled")
        m8195A.set_enabled(1,True)
        m8195A.set_enabled(2,False)
        self.assertEqual(m8195A.get_enabled(1),'1\n')
        self.assertEqual(m8195A.get_enabled(2),'0\n')

        print("Testing set analogHigh")
        m8195A.set_analog_high(3,0.5)
        self.assertEqual(m8195A.get_analog_high(3),0.5)

        print("Testing set analoLow")
        m8195A.set_analog_low(3,-0.5)
        self.assertEqual(m8195A.get_analog_low(3),-0.5)

        print("Testing set offset")
        m8195A.set_offset(1,0.05)
        self.assertEqual(float(m8195A.get_offset(1)),0.05)
