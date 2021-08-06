from typing import Union

from entropylab_qpudb import Resolver


class QPUResolver(Resolver):
    def __init__(self, aliases: Union[dict, None] = None):

        super().__init__(aliases)

    def q(self, qubit, channel=None) -> str:
        if channel == 'xy':
            return f'q{self._aliases.get(qubit, qubit)}_xy'
        elif channel == 'z':
            return f'q{self._aliases.get(qubit, qubit)}_z'
        elif channel is None:
            return f'q{self._aliases.get(qubit, qubit)}'
        elif channel:
            raise ValueError(f"channel {channel} is specified but not xy, z")

    def res(self, qubit) -> str:
        return f'r{self._aliases.get(qubit, qubit)}_1'

    def res_mixer(self, qubit):
        return f'mixer_ro'

    def q_mixer(self, qubit):
        return f'mixer_q{qubit}_xy'

    def feed_line(self, rr):
        # feed line 0 for rr 1-10
        # feed line 1 for rr 11-20
        return int((rr-1)/10)

    def coupler(self, qubit1, qubit2):
        pass



_qubit_aliases = {'q0': 1,
                  'q1': 2,
                  'q2': 3,
                  'q3': 4,
                  'q4': 5,
                  'q5': 6,
                  'q6': 7,
                  'q7': 8,
                  'q8': 9,
                  'q9': 10}

resolve = QPUResolver(_qubit_aliases)