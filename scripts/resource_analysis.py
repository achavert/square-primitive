import sys
import json
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


circuits = ['direct_inversion', 'optimized']
exact_counts = {'direct_inversion': {
                    'nqubits': [1.0, 4.0],
                    'X': [1.0, 6.0],
                    'CNOT': [-14.0, 14.0, 7.0],
                    'TOFFOLI': [-3.0, 6.0, 2.0]},
                'optimized': {
                    'nqubits': [2.0, 2.0],
                    'X': [-1.0, 4.0],
                    'CNOT': [-16.0, 14.5, 3.5],
                    'TOFFOLI': [-2.0, 3.0, 1.0]}}


data_labels = ["nqubits", "depth"]
gate_labels = ["X", "CNOT", "TOFFOLI"]


marker = 'x'


file = PROJECT_ROOT / "results" / "resources_m2-50.json"


def poly(_x, *a):
    return sum(coeff * np.asarray(_x)**i for i, coeff in enumerate(a))

def fit(x, y, degree = 1):
    param, _ = curve_fit(poly, x, y, p0=np.ones(degree + 1))
    return param

def polynomial_label(param):
    terms = []

    for i, coeff in enumerate(param):
        if np.isclose(coeff, 0):
            continue

        if i == 0:
            term = f'{abs(coeff):0.1f}'
        else:
            variable = 'm' if i == 1 else f'm^{i}'
            coefficient = '' if np.isclose(abs(coeff), 1) else f'{abs(coeff):0.1f}'
            term = coefficient + variable

        if not terms:
            terms.append(('-' if coeff < 0 else '') + term)
        else:
            sign = '-' if coeff < 0 else '+'
            terms.append(f' {sign} {term}')

    return ''.join(terms) if terms else '0'


def plot_fit(m, param, name):
    x_range = np.linspace(min(m), max(m), 1000)
    y = poly(x_range, *param)
    text = polynomial_label(param)
    line, = plt.plot(x_range, y, label = f'{name} fit: {text}') 
    return line

def plot_exact(m, label, name, color):
    param = exact_counts[name][label]
    y = poly(m, *param)
    text = polynomial_label(param)
    plt.scatter(m, y, color = color, marker=marker, label = f'{name} exact: {text}')

def generate_plot_nqubits(results, results_dir):
    for name in circuits:
        m = results[name]["m"]
        nqubits = results[name]["nqubits"]
        param = fit(m, nqubits)
        line = plot_fit(m, param, name)
        plot_exact(m, "nqubits", name, line.get_color())
    plt.title('Number of qubits scaling')
    plt.xlabel(r'$m$ (input bit length)')
    plt.ylabel(f'Total qubits')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    output_path = results_dir / f"nqubits.png"
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

def generate_plot_depth(results, results_dir):
    for name in circuits:
        m = results[name]["m"]
        nqubits = results[name]["depth"]
        param = fit(m, nqubits, degree=2)
        line = plot_fit(m, param, name)
    plt.title('Depth scaling')
    plt.xlabel(r'$m$ (input bit length)')
    plt.ylabel(f'Depth')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    output_path = results_dir / f"depth.png"
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def generate_plot_gate(results, gate, results_dir):
    if gate=='X': 
        degree = 1
    else:
        degree = 2
    for name in circuits:
        m = results[name]["m"]
        ngates = [results[name]["by_type"][k][gate] for k in range(len(m))]
        param = fit(m, ngates, degree=degree)
        line = plot_fit(m, param, name)
        plot_exact(m, gate, name, line.get_color())
    plt.title(f'{gate} gate scaling')
    plt.xlabel(r'$m$ (input bit length)')
    plt.ylabel(f'{gate} count')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    output_path = results_dir / f"{gate}_count.png"
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()



if __name__ == "__main__":
    with open(file, "r") as f:
        results = json.load(f)

    results_dir = PROJECT_ROOT / "results" / "graphs"
    results_dir.mkdir(parents=True, exist_ok=True)

    generate_plot_nqubits(results, results_dir)
    generate_plot_depth(results, results_dir)
    for gate in gate_labels:
        generate_plot_gate(results, gate, results_dir)





    
    
    #output_path = results_dir / f"resources_m{m_min}-{m_max}.json"
    
    
    