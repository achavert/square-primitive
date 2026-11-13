import sys
import json
from tqdm import tqdm
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import time
import numpy as np
import matplotlib.pyplot as plt
import quantum_squaring.circuits.square_optimized as square_optimized
import quantum_squaring.circuits.square_direct_inversion as square_direct_inversion

circuits = ['direct_inversion', 'optimized']

def build_circuit(m, name):
    if name == 'direct_inversion':
        _circ, _ = square_direct_inversion.Square(m, input_swaps = False, method = 'ripple_no_input_carry')
    elif name == 'optimized':
        _circ, _ = square_optimized.Square(m, input_swaps = False)
    return _circ

results = {name: {"m": [], "t": [], "t_dt": []} for name in circuits}

def generate_plot_times(results, graph_dir):
    for name in circuits:
        m = results[name]["m"]
        t = results[name]["t"]
        t_dt = results[name]["t_dt"]
        plt.errorbar(m, t, yerr=t_dt, markersize = 6, capsize=4, fmt=".-", label = f'{name}')


    plt.title(f'Simulation time scaling')
    plt.xlabel(r'$m$ (input bit length)')
    plt.ylabel(f'Wallclock time [s]')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    output_path = graph_dir / f"simulation_time.png"
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    m_min = 2
    m_max = 10
    reps = 10
    warm_up_reps = 2
    max_time = 1.0

    for name in circuits:
        walkout = False
        print(f'Circuit: {name}')
        for m in tqdm(range(m_min, m_max+1)):
            circ = build_circuit(m, name)
            for r in range(warm_up_reps):
                circ()
                times = []
            for r in range(reps):
                start = time.time()
                circ()
                stop = time.time()
                tim = stop - start
                if (tim) > max_time:
                    print(f'\n Simulation max time of {max_time:0.2f} s reached. Stopped at m={m}/{m_max}.')
                    walkout = True
                    break
                times.append(tim)
            if walkout is True:
                break
            else:
                t = np.mean(times)
                t_dt = np.std(times)
            res = results[name]
            res["m"].append(m); res["t"].append(t); res["t_dt"].append(t_dt)
    
    results_dir = PROJECT_ROOT / "results"
    graph_dir = PROJECT_ROOT / "results" / "graphs"
    results_dir.mkdir(parents=True, exist_ok=True)
    generate_plot_times(results, graph_dir)

    output_path = results_dir / f"simulation_time_m{m_min}-{m_max}_reps{reps}.json"

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"\nResults saved to: {output_path}")

