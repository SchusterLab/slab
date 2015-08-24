python -x %0 %*    &goto :eof

import os, sys
sys.path.append(r'C:\_Lib')
import slab
os.chdir(r'C:\_Lib\slab\cryomonitor')
execfile(r'C:\_Lib\slab\cryomonitor\gui_template.py')