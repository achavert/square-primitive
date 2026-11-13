import time
import sys

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import qibo
import quantum_squaring.circuits.square_optimized as square_optimized

qibo.set_backend("qibojit", platform="cuda")


def run_circ(m, output_file):
    num_qubits = 2*m + 2 # 
    start = time.time()
    for x in range(2**m):
        sq, (R, F) = square_optimized.Square(m, input_swaps = False)
        circ = square_optimized.initialize_square(x, m, allocs=(R, F), direct=True)
        circ.add(sq.on_qubits(*range(num_qubits)))
        circ.add(qibo.gates.M(*range(num_qubits)))

        res = circ(nshots = 1)

        samples = res.samples()[0]
        square_bits = samples[R[0]:R[2*m-1]+1]
        square_value = sum(bit << i for i, bit in enumerate(square_bits))
        print(
            f'Input: {x} | Result: {square_value} = {samples} | '
            f'res==x^2 {int(x**2) == int(square_value)}',
            file = output_file)
    end = time.time()
    print(
        f'Mean elapsed time: {(end-start)/2**m:0.4f} s\n',
        file = output_file )


if __name__ == "__main__":
    m_min = 2
    m_max = 10

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / f"validation_m{m_min}-{m_max}.txt"

    with output_path.open("w", encoding="utf-8") as output_file:
        for m in range(m_min, m_max+1):
            print("====================================", file = output_file)
            print(f"Input: m={m} bits", file = output_file)
            run_circ(m, output_file)
    print(f"Results saved to: {output_path}")

