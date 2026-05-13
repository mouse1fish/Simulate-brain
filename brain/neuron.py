from dataclasses import dataclass, field


@dataclass
class Neuron:
    id: int
    neuron_type: str = 'excitatory'
    potential: float = 0.0
    threshold: float = 1.0
    decay_rate: float = 0.95
    refractory_period: int = 3
    refractory_counter: int = 0
    fired: bool = False
    storage: bytearray = field(default_factory=lambda: bytearray(2048))
    fire_count: int = 0
    last_fire_tick: int = -1000

    @property
    def is_excitatory(self) -> bool:
        return self.neuron_type == 'excitatory'

    @property
    def is_inhibitory(self) -> bool:
        return self.neuron_type == 'inhibitory'

    @property
    def in_refractory(self) -> bool:
        return self.refractory_counter > 0

    def store_data(self, data: bytes, offset: int = 0):
        end = min(offset + len(data), 2048)
        self.storage[offset:end] = data[:end - offset]

    def read_data(self, offset: int = 0, length: int = 2048) -> bytes:
        end = min(offset + length, 2048)
        return bytes(self.storage[offset:end])

    def __repr__(self):
        state = 'FIRE!' if self.fired else ('refractory' if self.in_refractory else f'{self.potential:.3f}')
        return (f"Neuron({self.id}, {self.neuron_type[:3]}, "
                f"V={state}, fires={self.fire_count})")
