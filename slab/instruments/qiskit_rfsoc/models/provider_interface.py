"""
provider_interface.py
"""

import requests

from qiskit.providers import ProviderV1 as ProviderInterface

class SLabProviderInterface(ProviderInterface):
    """
    Fetch a backend interface through this class.

    References:
    [0] https://github.com/Qiskit/qiskit-ibmq-provider/blob/master/
        qiskit/providers/ibmq/accountprovider.py#L43
    """
    backends_url_dict = {
        "RFSoC2": "192.168.14.184:8555",
    }
    
    def __init__(self, timeout=20):
        """
        Args:
        timeout :: int - wait time in seconds for establishing
                         contact with a potential backend
        """
        super().__init__()
        self._backends = self._discover_backends(timeout)
    #ENDDEF

    def _discover_backends(self, timeout):
        """
        Discover remote backends.
        """
        backends = {}
        for key in self.backends_url_dict.keys():
            url = self.backends_url_dict[key]
            # attempt HEAD request to backend
            res = requests.head(url)
            try:
                res.raise_for_status()
                success = True
            except Exception as e:
                print("Could not establish contact with backend {} at {}.\n{}"
                      "".format(key, url, e))
                success = False
            #ENDTRY
            # if HEAD succeeded, expose the backend
            if success:
                backend = SLabBackendInterface(self, url)
                backends[backend.name()] = backend
            #ENDIF
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
