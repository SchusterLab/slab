__author__ = 'Aman LaChapelle'

from urllib.request import urlopen
from math import *
from slab import *
from slab.instruments import SocketInstrument

class MCRCA18(SocketInstrument):

    def __init__(self, name = "MCRCA18", address = "192.168.14.138:80"):
        SocketInstrument.__init__(self, name = name, address = '192.168.14.138:80')
        self.tcp_ip = address.split(":")[0]
        self.tcp_port = int(address.split(":")[1])

    def send_command(self, command = None, show = True):
        response = urlopen('http://%s:%d/%s' % (self.tcp_ip, self.port, command))
        if show == True:
            print(response.read())

    def get_largest(self):
        counter = []
        for i in range(7, 0, -1):
            try:
                ports_open = int(log(self.ports_copy, 2))
                counter.append(ports_open)
            except ValueError:
                pass
        return counter[0]

    def ports_one(self):
        response = urlopen('http://%s:%d/SWPORT?' % (self.tcp_ip, self.port))
        ports = int(response.read())
        if ports == 0:
            return "None" # edit this for the new array
        if ports != 0:
            ports_open = []
            possible = {0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F', 6:'G', 7:'H'}
            open_ports = []
            self.ports_copy = ports
            while self.ports_copy != 0:
                largest = self.get_largest()
                open_ports.append(largest)
                self.ports_copy -= 2**largest
            for i in range(0, len(open_ports)):
                ports_open.append(possible[open_ports[i]])
            return ports_open

    def switch_port(self, ports):
        ports_open = self.ports_one()
        for port in ports:
            if port in ports_open:
                self.send_command("SET%s=0" % port, show = False)
            else:
                self.send_command("SET%s=1" % port, show = False)
        return 0

    def all_to_zero(self):
        tmp = self.ports_one()
        if tmp == "None":
            return 0
        else:
            self.switch_port(tmp)
            return 0

    # Will need to change these as the diagram changes to make sure the switching still works.
    # As in lab notebook, cavities indexed 0-7 not 1-8
    def measure_cavity(self, number):
        binary_rep = "{0:03b}".format(number)

        possible = {'000':"", '001':"E", '010':"F", '011':["B", "F"], '100':"A", '101':["A", "D"],\
                    '110':["A", "C"], '111':["A", "C", "G"]}

        self.all_to_zero()
        self.switch_port(possible[binary_rep])





if __name__ == "__main__":
    print("Usage:\n>>> <name> = rf_switch()\n>>> " \
          "<name>.ports_one(<port>)\n>>> <name>.switch_port(<port>)" \
          "\n>>> <name>.send_command('<command>')")
    print("Commands are here:\n" \
          "http://www.minicircuits.com/softwaredownload/Prog_Manual-2-Switch.pdf\n" \
          "Section 4.4")
    temp = MCRCA18()
    print(temp.ports_one())
    temp.measure_cavity(3)
    print(temp.ports_one())
    temp.all_to_zero()
    #command = raw_input("Enter a command: ")
    #temp.send_command(command, show = False)














