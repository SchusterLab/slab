from distutils.core import setup
import py2exe
windows = [{'script':'launcher.py', 'icon_resources':[(0, 'launcher_icon.ico')]}]
#setup(console=['launcher.py'], options={'py2exe':{'includes':['sip']}})
setup(windows=windows, options={'py2exe':{'includes':['sip']}})