# -*- coding: utf-8 -*-
"""
"""

from slab.instruments import InstrumentManager
from liveplot import LivePlotClient


im = InstrumentManager()

#nwa = im['PNAX2']
bk = im['bkp_bf2']
print bk.get_id()