import sys
import json
from tqdm import tqdm
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from collections import Counter
import quantum_squaring.circuits.square_optimized as square_optimized
import quantum_squaring.circuits.square_direct_inversion as square_direct_inversion

circuits = ['direct_inversion', 'optimized']



def build_circuit(m, name):
    if name == 'direct_inversion':
        _circ, _ = square_direct_inversion.Square(m, input_swaps = False, method = 'ripple_no_input_carry')
    elif name == 'optimized':
        _circ, _ = square_optimized.Square(m, input_swaps = False)
    return _circ

results = {name: {"m": [], "nqubits": [], "gates": [], "by_type": [], "depth":[]} for name in circuits}


if __name__ == "__main__":
    m_min = 2
    m_max = 50
    for name in circuits:
        print(f'Circuit: {name}')
        for m in tqdm(range(m_min, m_max+1)):
            circ = build_circuit(m, name)
            by_type = Counter(type(g).__name__ for g in circ.queue)
            r = results[name]
            r["m"].append(m); r["gates"].append(len(circ.queue)); r["by_type"].append(dict(by_type)); r["depth"].append(circ.depth); r["nqubits"].append(circ.nqubits)
    
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    output_path = results_dir / f"resources_m{m_min}-{m_max}.json"

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"\nResults saved to: {output_path}")