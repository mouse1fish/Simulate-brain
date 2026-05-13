import sys
import time
import numpy as np
from brain.brain import Brain


def clear_screen():
    sys.stdout.write('\033[2J\033[H')
    sys.stdout.flush()


def activity_char(rate):
    if rate <= 0:
        return ' '
    elif rate < 0.1:
        return '·'
    elif rate < 0.3:
        return '○'
    elif rate < 0.6:
        return '●'
    else:
        return '█'


def potential_char(pot, max_pot=1.0):
    ratio = min(pot / max_pot, 1.0) if max_pot > 0 else 0
    if ratio <= 0:
        return ' '
    elif ratio < 0.2:
        return '░'
    elif ratio < 0.4:
        return '▒'
    elif ratio < 0.6:
        return '▓'
    elif ratio < 0.8:
        return '█'
    else:
        return '◆'


def print_header(brain):
    status = brain.get_status()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║              🧠  Brain Simulator  -  1000 Neurons          ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print(f"║  Tick: {status['tick']:>6d}  │  Active: {status['active']:>4d}/{status['neurons']}  │  "
          f"Fires: {status['total_fires']:>8d}  │  Learn: {'ON ' if status['learning'] else 'OFF'}  ║")
    print(f"║  Avg V: {status['avg_potential']:.4f}  │  Rate: {status['avg_fire_rate']:.6f}  │  "
          f"Exc: {status['excitatory']}  Inh: {status['inhibitory']}  ║")
    print("╚══════════════════════════════════════════════════════════════╝")


def print_region_activity(brain):
    bars = brain.get_region_activity_bar()
    print("\n┌─ Brain Region Activity ─────────────────────────────────────┐")
    for name in ['visual', 'auditory', 'motor', 'association', 'memory']:
        if name in bars:
            print(f"│  {bars[name]}  │")
    print("└─────────────────────────────────────────────────────────────┘")


def print_activity_heatmap(brain, cols=60):
    activity = brain.get_activity_map(cols)
    print("\n┌─ Neural Activity Heatmap ──────────────────────────────────┐")
    line = "│  "
    for rate in activity:
        line += activity_char(rate)
    line += "  │"
    print(line)
    print("│  " + "VIS" + " " * 17 + "AUD" + " " * 8 + "MOT" + " " * 8 +
          "ASC" + " " * 18 + "MEM" + "  │")
    print("└─────────────────────────────────────────────────────────────┘")


def print_potential_heatmap(brain, cols=60):
    potentials = brain.get_potential_map(cols)
    max_pot = max(potentials) if potentials else 1.0
    max_pot = max(max_pot, 0.001)
    print("\n┌─ Membrane Potential Map ───────────────────────────────────┐")
    line = "│  "
    for pot in potentials:
        line += potential_char(pot, max_pot)
    line += "  │"
    print(line)
    print(f"│  Max potential: {max_pot:.4f}                                       │")
    print("└─────────────────────────────────────────────────────────────┘")


def print_activity_graph(brain, width=50):
    history = brain.activity_history[-width:]
    if not history:
        return
    print("\n┌─ Activity Over Time ───────────────────────────────────────┐")
    max_val = max(max(history), 0.01)
    height = 6
    for row in range(height - 1, -1, -1):
        threshold = max_val * (row + 1) / height
        line = "│  "
        for val in history:
            if val >= threshold:
                line += '█'
            elif val >= threshold - max_val / height:
                line += '▄'
            else:
                line += ' '
        line += "  │"
        print(line)
    print(f"│  {'─' * width}  │")
    print(f"│  Now{' ' * (width - 7)}Past  │")
    print("└─────────────────────────────────────────────────────────────┘")


def print_neuron_detail(brain, nid):
    try:
        info = brain.get_neuron_info(nid)
    except IndexError as e:
        print(f"  Error: {e}")
        return

    region = info['region']
    ntype = info['type']
    fired_str = "⚡ FIRE!" if info['fired'] else "quiet"
    ref_str = f"refractory({info['refractory']})" if info['refractory'] > 0 else "ready"

    print(f"\n┌─ Neuron #{nid} ─────────────────────────────────────────────┐")
    print(f"│  Type: {ntype:12s}  Region: {region:12s}               │")
    print(f"│  Potential: {info['potential']:.6f}  Threshold: {info['threshold']:.2f}          │")
    print(f"│  State: {fired_str:12s}  {ref_str:20s}        │")
    print(f"│  Fire count: {info['fire_count']:>6d}  Last fire: tick {info['last_fire']:>6d}    │")
    print(f"│  Out strength: {info['out_strength']:.4f}  In strength: {info['in_strength']:.4f}  │")

    data = brain.read_data(nid, 0, 64)
    non_zero = sum(1 for b in data if b != 0)
    print(f"│  Storage used: {non_zero}/2048 bytes                              │")
    if non_zero > 0:
        try:
            text = data.rstrip(b'\x00').decode('utf-8', errors='replace')[:40]
            print(f"│  Data preview: {text:<43s} │")
        except Exception:
            print(f"│  Data: (binary)                                            │")
    print("└─────────────────────────────────────────────────────────────┘")


def print_weight_stats(brain):
    stats = brain.get_weight_stats()
    print("\n┌─ Synaptic Weight Statistics ───────────────────────────────┐")
    print(f"│  Excitatory: mean={stats['exc_mean']:.6f}  max={stats['exc_max']:.6f}       │")
    print(f"│  Inhibitory: mean={stats['inh_mean']:.6f}  min={stats['inh_min']:.6f}       │")
    print(f"│  Connections: {stats['total_connections']:>10d}  Sparsity: {stats['sparsity']:.4f}       │")
    print("└─────────────────────────────────────────────────────────────┘")


def print_help():
    print("""
┌─ Available Commands ─────────────────────────────────────────┐
│                                                              │
│  tick [n]              Run n simulation ticks (default: 1)   │
│  auto [n] [delay]      Auto-run n ticks with display         │
│  stim <id> [strength]  Stimulate a neuron                   │
│  region <name> [str]   Stimulate a brain region              │
│  pattern <name>        Inject a named pattern                │
│  status                Show brain status                     │
│  neuron <id>           Show neuron details                   │
│  heatmap               Show activity heatmap                │
│  potential             Show potential map                    │
│  graph                 Show activity over time               │
│  weights               Show weight statistics                │
│  learn [on|off]        Toggle Hebbian learning               │
│  store <id> <text>     Store text in neuron (2KB max)       │
│  read <id> [len]       Read neuron storage                   │
│  reset                 Reset brain state                     │
│  help                  Show this help                        │
│  quit                  Exit simulator                        │
│                                                              │
│  Brain Regions: visual, auditory, motor, association, memory │
│  Patterns: light, flash, sound, noise, move, kick,          │
│            think, ponder, remember, recall                   │
└──────────────────────────────────────────────────────────────┘
""")


def display_full(brain):
    clear_screen()
    print_header(brain)
    print_region_activity(brain)
    print_activity_heatmap(brain)
    print_potential_heatmap(brain)
    print_activity_graph(brain)


def run_auto(brain, n_ticks=50, delay=0.1):
    for i in range(n_ticks):
        brain.tick()
        if (i + 1) % 5 == 0 or i == 0:
            display_full(brain)
            print(f"\n  Auto-running... tick {i+1}/{n_ticks}  (Ctrl+C to stop)")
            time.sleep(delay)


def main():
    print("\nInitializing brain with 1000 neurons...")
    brain = Brain(num_neurons=1000, avg_connections=1000, learning_rate=0.005, seed=42)
    print("Building synaptic connections...")
    print(f"  {brain.num_neurons} neurons created")
    print(f"  {int(np.sum(brain.is_excitatory))} excitatory, {int(np.sum(brain.is_inhibitory))} inhibitory")
    print(f"  {brain.num_neurons * (brain.num_neurons - 1)} synaptic connections")
    print(f"  {brain.num_neurons * 2}KB total neuron storage")
    print("Brain ready!\n")

    display_full(brain)
    print_help()

    while True:
        try:
            cmd = input("\n🧠 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not cmd:
            continue

        parts = cmd.split()
        action = parts[0].lower()

        try:
            if action == 'quit' or action == 'exit':
                print("Goodbye!")
                break

            elif action == 'help' or action == '?':
                print_help()

            elif action == 'tick' or action == 't':
                n = int(parts[1]) if len(parts) > 1 else 1
                for _ in range(n):
                    brain.tick()
                display_full(brain)
                active = int(np.sum(brain.fired))
                print(f"\n  Ran {n} tick(s). Active neurons: {active}")

            elif action == 'auto' or action == 'a':
                n = int(parts[1]) if len(parts) > 1 else 50
                delay = float(parts[2]) if len(parts) > 2 else 0.1
                try:
                    run_auto(brain, n, delay)
                except KeyboardInterrupt:
                    print("\n  Auto-run stopped.")

            elif action == 'stim' or action == 's':
                if len(parts) < 2:
                    print("  Usage: stim <neuron_id> [strength]")
                    continue
                nid = int(parts[1])
                strength = float(parts[2]) if len(parts) > 2 else 2.0
                brain.stimulate(nid, strength)
                print(f"  Stimulated neuron {nid} with strength {strength}")

            elif action == 'region' or action == 'r':
                if len(parts) < 2:
                    print("  Regions: " + ", ".join(Brain.REGIONS.keys()))
                    continue
                region_name = parts[1].lower()
                strength = float(parts[2]) if len(parts) > 2 else 2.0
                count = brain.stimulate_region(region_name, strength)
                if count > 0:
                    print(f"  Stimulated {count} neurons in '{region_name}' region (strength={strength})")
                else:
                    print(f"  Unknown region: {region_name}")
                    print("  Available: " + ", ".join(Brain.REGIONS.keys()))

            elif action == 'pattern' or action == 'p':
                if len(parts) < 2:
                    print("  Patterns: " + ", ".join(Brain.PATTERNS.keys()))
                    continue
                pattern_name = parts[1].lower()
                count = brain.stimulate_pattern(pattern_name)
                if count > 0:
                    region, strength = Brain.PATTERNS[pattern_name]
                    print(f"  Injected pattern '{pattern_name}' -> {region} region (strength={strength}, {count} neurons)")
                else:
                    print(f"  Unknown pattern: {pattern_name}")
                    print("  Available: " + ", ".join(Brain.PATTERNS.keys()))

            elif action == 'status':
                display_full(brain)
                print_weight_stats(brain)

            elif action == 'neuron' or action == 'n':
                if len(parts) < 2:
                    print("  Usage: neuron <id>")
                    continue
                nid = int(parts[1])
                print_neuron_detail(brain, nid)

            elif action == 'heatmap':
                print_activity_heatmap(brain)

            elif action == 'potential':
                print_potential_heatmap(brain)

            elif action == 'graph':
                print_activity_graph(brain)

            elif action == 'weights' or action == 'w':
                print_weight_stats(brain)

            elif action == 'learn':
                if len(parts) > 1:
                    if parts[1].lower() == 'on':
                        brain.learning_enabled = True
                    elif parts[1].lower() == 'off':
                        brain.learning_enabled = False
                else:
                    brain.learning_enabled = not brain.learning_enabled
                state = "ON" if brain.learning_enabled else "OFF"
                print(f"  Hebbian learning: {state}")

            elif action == 'store':
                if len(parts) < 3:
                    print("  Usage: store <neuron_id> <text>")
                    continue
                nid = int(parts[1])
                text = ' '.join(parts[2:])
                brain.store_data(nid, text)
                print(f"  Stored {len(text.encode('utf-8'))} bytes in neuron {nid}")

            elif action == 'read':
                if len(parts) < 2:
                    print("  Usage: read <neuron_id> [length]")
                    continue
                nid = int(parts[1])
                length = int(parts[2]) if len(parts) > 2 else 256
                data = brain.read_data(nid, 0, length)
                non_zero = sum(1 for b in data if b != 0)
                if non_zero > 0:
                    try:
                        text = data.rstrip(b'\x00').decode('utf-8', errors='replace')
                        print(f"  Neuron {nid} storage ({non_zero} bytes used):")
                        print(f"  \"{text}\"")
                    except Exception:
                        print(f"  Neuron {nid}: {non_zero} bytes (binary data)")
                else:
                    print(f"  Neuron {nid}: storage empty")

            elif action == 'reset':
                brain.reset()
                print("  Brain state reset.")

            else:
                print(f"  Unknown command: {action}. Type 'help' for commands.")

        except ValueError as e:
            print(f"  Invalid argument: {e}")
        except IndexError as e:
            print(f"  Error: {e}")
        except Exception as e:
            print(f"  Error: {e}")


if __name__ == '__main__':
    main()
