"""
provider_interface.py
"""

from qiskit.providers import ProviderV1 as ProviderInterface

class SLabProviderInterface(ProviderInterface):
    """
    Fetch a backend interface through this class.

    References:
    [0] https://github.com/Qiskit/qiskit-ibmq-provider/blob/master/
        qiskit/providers/ibmq/accountprovider.py#L43
    """
    backends_ip_dict = {
        "RFSoC2": "192.168.14.184",
    }
    
    def __init__(self, timeout=20):
        """
        Args:
        timeout :: int - wait time in seconds for establishing
                         contact with a potential backend
        """
        super().__init__()
        self._backends = self._backends(timeout)
    #ENDDEF

    def _backends(self, timeout):
        """
        Discover remote backends.
        """
        backends = {}
        for key in self.backends_ip_dict.keys():
            ip = self.backends_ip_dict[key]
            
            backend = SLabBackendInterface(configuration, defaults, self, ip)
            backends[backend.name()] = backend
        #ENDFOR
        return backends
    #ENDDEF

    def backends(self, **kwargs):
        return self._backends
    #ENDDEF

    def get_backend(self, name):
        backend = self._backends.get(name, None)
        return backend
    #ENDDEF
#ENDCLASS
