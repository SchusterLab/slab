import os
try: import instruments
except: print "Warning: Could not import instruments"
try: from dsfit import *
except: print "Warning: Could not import fitting package"
try: from dataanalysis import *
except: print "Warning: Could not inmport dataanalysis"
try: from plotting import *
except: print "Warning: Could not import plotting"
try: import gui
except: print "Warning: Could not import gui"
try: import script
except: print "Warning: Could not import script"
try: from analysisScript import *
except: print "Warning: Could not install analysisScript"
from datamanagement import *
try: from circuitqed import *
except: print "Warning: Could not import circuitqed. Probably due to qutip"
#import diamond
try: import widgets
except: print "Warning: Could not import widgets"