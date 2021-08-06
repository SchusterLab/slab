from slab.instruments import WebInstrument
from urllib.request import urlopen

class MiniCircuitsSwitch(WebInstrument):
    def __init__(self, name, address='', enabled=True):
        super().__init__(name=name, address=address, enabled=enabled)
        if self.address[-1] != "/":
            self.address += "/"
        self.model = self.get_model()
        self.serial = self.get_serial()

    def write(self, s):
        result = urlopen(self.address + ":" + s)
        return result

    def query(self, s):
        result = urlopen(self.address + ":" + s)
        return result.read()

    def get_serial(self):
        ans = self.query("SN?")
        return int(ans[3:])

    def get_model(self):
        ans = self.query("MN?")
        return ans[3:].decode()

    def get_id(self):
        return f"Minicircuits RF-Switch: {self.address}, model: {self.model}, serial: {self.serial}"

    def get_switches(self):
        binnum = [int(i) for i in bin(int(self.query("SWPORT?")))[2:]]
        binnum.reverse()
        return binnum

    def get_switch(self, ch):
        return self.get_switches()[ch]

    def set_switch(self, ch, val):
        chabc = chr(65 + ch)
        cmd = f"SET{chabc}={val}"
        print(cmd)
        self.write(cmd)

    def set_all(self, val):
        old_vals = self.get_switches()
        for ch in range(len(old_vals)):
            self.set_switch(ch, val)