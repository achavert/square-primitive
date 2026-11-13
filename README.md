# Quantum squaring primitive

Qibo implementation of the in-place, qubit-optimized and garbageless squaring operator based on the inversion of the non-restoring square-root algorithm from Muñoz-Coreas & Thapliyal, with validation and resource analysis.

# Setup

```
python -m venv .square_venv
source .square_venv/bin/activate
pip install -r requirements.txt
```
# Contents


| File | Summary |
|------|------|
| `src/quantum_squaring/circuits/square_optimized.py` | (Main) Proposed squaring primitive: 2m+2 qubits for an m-bit input |
| `src/quantum_squaring/circuits/square_direct_inversion.py` | Direct inversion of the corrected Muñoz-Coreas & Thapliyal: 4m+1 qubits for an m-bit input |
| `src/quantum_squaring/circuits/square_qubit_minimized.py` | Qubit minimized squaring circuit (with C3X gates): 2m+1 qubits for an m-bit input |
| `src/quantum_squaring/circuits/square_direct_inversion_no_anc.py` | Direct inversion of the corrected Muñoz-Coreas & Thapliyal without ancilla control: 4m qubits for an m-bit input |
|------|------|
| `scripts/exhaustive_verification.py` | Exhaustive validation of the main square primitive by classical simulation |
| `scripts/resource_data_generation.py` | Generation of qubit, gate and depth counts for the main proposal and the direct inversion. Sweep of the input length m |
| `scripts/resource_analysis.py` | Validation of the exact derived counts for the main proposal and the direct inversion circuit's qubit, gate and depth scalings by fitting the experimental data |
| `scripts/simulation_time_analysis.py` | Evaluation of the GPU simulation time of the main proposal and the direct inversion circuits. Sweep of the input length m |
