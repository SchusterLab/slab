from slab.instruments import InstrumentManager
#from slab.instruments.voltsource import YokogawaGS200
# don't think i need the above because i already imported as YOKO1 thru instrumentmanager
im = InstrumentManager()

yigbias = im['YOKO1']
levell = yigbias.get_level()
print(levell)
#yigbias.set_level(0.160)
#print(yigbias.get_level())
yigbias.set_level(0.165)
print(yigbias.get_level())