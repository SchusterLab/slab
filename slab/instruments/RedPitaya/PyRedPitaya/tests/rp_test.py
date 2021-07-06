import numpy as np
from PyRedPitaya.instrument import RedPitaya


def check_conv_buffer_readwrite(rp):
    ans = True
    for offset in [0x30000, 0x40000, 0x50000, 0x60000]:
        data = [rp.scope.from_pyint(v) for v in np.random.randint(0, 2 ** 13, 2 ** 12)]
        rp.scope.writes(offset, data)
        rdata = np.array([rp.scope.to_pyint(v) for v in rp.scope.reads(offset, len(data))])
        ans = ans and np.array_equal(rdata, data)
    return ans

if __name__ == "__main__":
    rp = RedPitaya()

    check_conv_buffer_readwrite(rp)

