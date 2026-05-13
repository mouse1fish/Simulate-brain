import numpy as np
from .neuron import Neuron


class Brain:
    REGIONS = {
        'visual': (0, 200),
        'auditory': (200, 350),
        'motor': (350, 500),
        'association': (500, 800),
        'memory': (800, 1000),
    }

    REGION_LABELS = {
        'visual': 'VIS',
        'auditory': 'AUD',
        'motor': 'MOT',
        'association': 'ASC',
        'memory': 'MEM',
    }

    PATTERNS = {
        'light': ('visual', 0.5),
        'flash': ('visual', 2.0),
        'sound': ('auditory', 0.5),
        'noise': ('auditory', 2.0),
        'move': ('motor', 0.5),
        'kick': ('motor', 2.0),
        'think': ('association', 0.5),
        'ponder': ('association', 2.0),
        'remember': ('memory', 0.5),
        'recall': ('memory', 2.0),
    }

    def __init__(self, num_neurons=1000, avg_connections=1000,
                 learning_rate=0.005, seed=42):
        self.rng = np.random.default_rng(seed)
        self.num_neurons = num_neurons
        self.avg_connections = min(avg_connections, num_neurons - 1)
        self.learning_rate = learning_rate
        self.tick_count = 0
        self.learning_enabled = True

        n_excitatory = int(num_neurons * 0.8)
        self.is_excitatory = np.zeros(num_neurons, dtype=bool)
        exc_indices = self.rng.choice(num_neurons, n_excitatory, replace=False)
        self.is_excitatory[exc_indices] = True
        self.is_inhibitory = ~self.is_excitatory

        self.potentials = self.rng.uniform(0, 0.05, num_neurons).astype(np.float64)
        self.thresholds = np.ones(num_neurons, dtype=np.float64)
        self.decay_rates = np.full(num_neurons, 0.95, dtype=np.float64)
        self.refractory_counters = np.zeros(num_neurons, dtype=np.int32)
        self.refractory_periods = np.full(num_neurons, 3, dtype=np.int32)
        self.fired = np.zeros(num_neurons, dtype=bool)

        self.weights = np.zeros((num_neurons, num_neurons), dtype=np.float64)
        n_exc = int(np.sum(self.is_excitatory))
        n_inh = int(np.sum(self.is_inhibitory))
        exc_cols = np.where(self.is_excitatory)[0]
        inh_cols = np.where(self.is_inhibitory)[0]
        for j in exc_cols:
            self.weights[:, j] = self.rng.uniform(0, 0.03, num_neurons)
        for j in inh_cols:
            self.weights[:, j] = self.rng.uniform(-0.06, 0, num_neurons)
        np.fill_diagonal(self.weights, 0)

        self.storage = np.zeros((num_neurons, 2048), dtype=np.uint8)

        self.fire_counts = np.zeros(num_neurons, dtype=np.int64)
        self.last_fire_time = np.full(num_neurons, -1000, dtype=np.int64)
        self.total_fires = 0
        self.activity_history = []
        self._neuron_cache = [None] * num_neurons

    def tick(self, external_input=None):
        self.tick_count += 1

        self.potentials *= self.decay_rates

        spikes = self.fired.astype(np.float64)
        if np.any(spikes):
            synaptic_input = self.weights @ spikes
            self.potentials += synaptic_input

        if external_input is not None:
            self.potentials += external_input

        was_in_refractory = self.refractory_counters > 0
        self.refractory_counters = np.maximum(self.refractory_counters - 1, 0)
        still_in_refractory = self.refractory_counters > 0

        can_fire = ~still_in_refractory
        should_fire = (self.potentials >= self.thresholds) & can_fire

        self.fired = should_fire
        self.potentials[should_fire] = 0.0
        self.refractory_counters[should_fire] = self.refractory_periods[should_fire]
        self.fire_counts[should_fire] += 1
        self.last_fire_time[should_fire] = self.tick_count
        self.total_fires += int(np.sum(should_fire))

        if self.learning_enabled and np.any(self.fired):
            self._learn()

        activity = float(np.mean(self.fired))
        self.activity_history.append(activity)
        if len(self.activity_history) > 2000:
            self.activity_history = self.activity_history[-1000:]

        return self.fired.copy()

    def _learn(self):
        fired = self.fired.astype(np.float64)
        recent = np.exp(-0.2 * (self.tick_count - self.last_fire_time).astype(np.float64))
        recent[self.last_fire_time < 0] = 0.0

        delta = self.learning_rate * np.outer(fired, recent)
        self.weights += delta

        self.weights *= 0.9999

        np.fill_diagonal(self.weights, 0)

        exc_cols = self.is_excitatory
        inh_cols = self.is_inhibitory
        self.weights[:, exc_cols] = np.maximum(self.weights[:, exc_cols], 0)
        self.weights[:, inh_cols] = np.minimum(self.weights[:, inh_cols], 0)

        np.clip(self.weights, -2.0, 2.0, out=self.weights)

    def stimulate(self, neuron_ids, strength=2.0):
        if isinstance(neuron_ids, (int, np.integer)):
            neuron_ids = [int(neuron_ids)]
        neuron_ids = np.asarray(neuron_ids)
        valid = neuron_ids < self.num_neurons
        self.potentials[neuron_ids[valid]] += strength

    def stimulate_region(self, region_name, strength=2.0):
        if region_name in self.REGIONS:
            start, end = self.REGIONS[region_name]
            self.potentials[start:end] += strength
            return end - start
        return 0

    def stimulate_pattern(self, pattern_name):
        if pattern_name in self.PATTERNS:
            region, strength = self.PATTERNS[pattern_name]
            return self.stimulate_region(region, strength)
        return 0

    def get_neuron(self, nid) -> Neuron:
        if nid < 0 or nid >= self.num_neurons:
            raise IndexError(f"Neuron ID {nid} out of range [0, {self.num_neurons})")
        n = self._neuron_cache[nid]
        if n is None:
            n = Neuron(id=nid)
            self._neuron_cache[nid] = n
        n.neuron_type = 'excitatory' if self.is_excitatory[nid] else 'inhibitory'
        n.potential = float(self.potentials[nid])
        n.threshold = float(self.thresholds[nid])
        n.refractory_counter = int(self.refractory_counters[nid])
        n.fired = bool(self.fired[nid])
        n.fire_count = int(self.fire_counts[nid])
        n.last_fire_tick = int(self.last_fire_time[nid])
        n.storage = bytearray(self.storage[nid].tobytes())
        return n

    def get_neuron_info(self, nid) -> dict:
        return {
            'id': nid,
            'type': 'excitatory' if self.is_excitatory[nid] else 'inhibitory',
            'region': self._get_region(nid),
            'potential': float(self.potentials[nid]),
            'threshold': float(self.thresholds[nid]),
            'fired': bool(self.fired[nid]),
            'fire_count': int(self.fire_counts[nid]),
            'refractory': int(self.refractory_counters[nid]),
            'last_fire': int(self.last_fire_time[nid]),
            'out_strength': float(np.sum(np.abs(self.weights[:, nid]))),
            'in_strength': float(np.sum(np.abs(self.weights[nid, :]))),
        }

    def _get_region(self, nid):
        for name, (start, end) in self.REGIONS.items():
            if start <= nid < end:
                return name
        return 'unknown'

    def get_status(self) -> dict:
        active = int(np.sum(self.fired))
        avg_pot = float(np.mean(self.potentials))
        avg_rate = float(self.total_fires / max(self.tick_count * self.num_neurons, 1))
        recent = self.activity_history[-20:] if self.activity_history else [0.0]

        region_activity = {}
        for name, (start, end) in self.REGIONS.items():
            region_activity[name] = {
                'active': int(np.sum(self.fired[start:end])),
                'total': end - start,
                'avg_potential': float(np.mean(self.potentials[start:end])),
                'fire_rate': float(np.sum(self.fire_counts[start:end]) /
                                   max(self.tick_count * (end - start), 1)),
            }

        return {
            'tick': self.tick_count,
            'neurons': self.num_neurons,
            'excitatory': int(np.sum(self.is_excitatory)),
            'inhibitory': int(np.sum(self.is_inhibitory)),
            'active': active,
            'total_fires': self.total_fires,
            'avg_potential': avg_pot,
            'avg_fire_rate': avg_rate,
            'recent_activity': recent,
            'learning': self.learning_enabled,
            'region_activity': region_activity,
        }

    def store_data(self, nid, data, offset=0):
        if isinstance(data, str):
            data = data.encode('utf-8')
        end = min(offset + len(data), 2048)
        actual_end = offset + len(data)
        if actual_end > 2048:
            data = data[:2048 - offset]
        data_array = np.frombuffer(data, dtype=np.uint8).copy()
        self.storage[nid, offset:offset + len(data_array)] = data_array

    def read_data(self, nid, offset=0, length=2048):
        end = min(offset + length, 2048)
        return bytes(self.storage[nid, offset:end])

    def get_activity_map(self, cols=50):
        chunk = max(self.num_neurons // cols, 1)
        n_chunks = (self.num_neurons + chunk - 1) // chunk
        result = []
        for i in range(n_chunks):
            start = i * chunk
            end = min(start + chunk, self.num_neurons)
            rate = float(np.mean(self.fired[start:end]))
            result.append(rate)
        return result

    def get_potential_map(self, cols=50):
        chunk = max(self.num_neurons // cols, 1)
        n_chunks = (self.num_neurons + chunk - 1) // chunk
        result = []
        for i in range(n_chunks):
            start = i * chunk
            end = min(start + chunk, self.num_neurons)
            avg_pot = float(np.mean(self.potentials[start:end]))
            result.append(avg_pot)
        return result

    def get_region_activity_bar(self):
        bars = {}
        for name, (start, end) in self.REGIONS.items():
            active = int(np.sum(self.fired[start:end]))
            total = end - start
            pct = active / total
            bar_len = 20
            filled = int(pct * bar_len)
            bar = '█' * filled + '░' * (bar_len - filled)
            label = self.REGION_LABELS.get(name, name[:3])
            bars[name] = f"{label} [{bar}] {active:3d}/{total} ({pct*100:5.1f}%)"
        return bars

    def get_weight_stats(self):
        exc_w = self.weights[:, self.is_excitatory]
        inh_w = self.weights[:, self.is_inhibitory]
        return {
            'exc_mean': float(np.mean(exc_w)),
            'exc_max': float(np.max(exc_w)),
            'inh_mean': float(np.mean(inh_w)),
            'inh_min': float(np.min(inh_w)),
            'total_connections': int(np.count_nonzero(self.weights)),
            'sparsity': 1.0 - float(np.count_nonzero(self.weights)) / (self.num_neurons ** 2),
        }

    def inject_signal(self, signal_array):
        if len(signal_array) != self.num_neurons:
            raise ValueError(f"Signal length {len(signal_array)} != {self.num_neurons}")
        self.potentials += np.asarray(signal_array, dtype=np.float64)

    def reset(self):
        self.potentials[:] = self.rng.uniform(0, 0.05, self.num_neurons)
        self.fired[:] = False
        self.refractory_counters[:] = 0
        self.tick_count = 0
        self.fire_counts[:] = 0
        self.last_fire_time[:] = -1000
        self.total_fires = 0
        self.activity_history.clear()
