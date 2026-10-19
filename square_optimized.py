''' Quantum squaring primitive (in-place) by inverting the non-restoring square root

Register layout, with n=2*m (m = number of bits of the input x):

    R: n qubits -> holds x**2 at the end
    F: n qubits -> "root" register, its logical order is permuted while the sqrt runs
    Z: 1 qubit  -> sing/control qubit

Total qubits: 2*n + 1 = 4*m +1 
'''
import numpy as np
import qibo

ADD_SUB_METHODS = ('ripple_input_carry', 'ripple_no_input_carry', 'aqft')




def Square_opt(m, allocs = None, input_swaps = False) -> qibo.Circuit:
    '''Builds the circuit that maps 
            |x>|0>...|0>    ->  |x**2>|0>...|0>
    Returns (circuit, (R,F,Z)). The returned F is the final layout of the sqrt, which is where the input must be loaded:

        - input_swaps=False: load x directly on F[2:m+2] from the returned allocs.
        - input_swaps=True: load x on R[0:m], the circuits moves it to F with m extra SWAPs

    The result x**2 is always on R[0:n], in little-endian configuration (least-significant bit on R[0]).     
    '''
    n = 2 * m
    nqubits = 2 * n + 1

    R, F, Z = get_allocs(n) if allocs is None else allocs
    sqrt_qc, (R, F, Z) = sqrt_circ(n, (R, F, Z))
    inv_sqrt_qc = sqrt_qc.invert()

    _qc = qibo.Circuit(nqubits)
    if input_swaps:
        for p in range(m):
            _qc.add(qibo.gates.SWAP(R[p], F[p+2]))
    _qc.add(qibo.gates.X(F[0]))
    _qc.add(inv_sqrt_qc.on_qubits(*range(nqubits)))
    _qc.add(qibo.gates.X(F[0]))
    return _qc, (R, F, Z)


def sqrt_circ(n, allocs, method = 'ripple_input_carry', **kargs) -> qibo.Circuit:
    ''' Non-restoring sqrt of an n-bit number. Returns (circuit, (R,F,Z)).

    The input 'allocs' is not modified, the returned F holds the final layout.
    '''
    R, F, Z = allocs; F = list(F)
    nqubits = 2 * n +1
    _qc_sqrt = qibo.Circuit(nqubits)

    initial_substraction(_qc_sqrt, n, (R, F, Z))
    for it in range(2, n//2):
        conditional_add_sub(_qc_sqrt, n, it, (R, F, Z))
    remainder_restorarion(_qc_sqrt, n, (R, F, Z))
    return _qc_sqrt, (R, F, Z)


def initialize_square(x, m, allocs = None, direct = False) -> qibo.Circuit:
    ''' Circuit that loads the m-bit integer x (little-endian):
            - direct=False: onto R[0:m] (use with input_swaps=True).
            - direct=True:  onto F[2:m+2] of the final layout, allocs must be the ones returned by Square
    '''
    if not 0 <= x < 2**m:
        raise ValueError(f"x={x} does not fit in {m} bits")
    if direct and allocs is None:
        raise ValueError("direct=True needs the allocs returned by Square")

    n = 2 * m; nqubits = 2*n + 1
    R, F, Z =get_allocs(n) if allocs is None else allocs
    register = F[2:m+2] if direct else R[:m]

    init_circ = qibo.Circuit(nqubits)
    for i, q in enumerate(register):
        if (x >> i) & 1:
            init_circ.add(qibo.gates.X(q))
    return init_circ
# ----------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------

def get_allocs(n):
    ''' Default initial allocations for the circuit:
    R: range(0, n)
    F: range(n, 2 * n)  -> prior to the permutations performed by Square
    Z: 2 * n
    '''
    R = list(range(0, n))             
    F = list(range(n, 2 * n))        
    Z = 2 * n    
    return R, F, Z    

def swap_allocs(_p, _q, _allocs):
    ''' Label reallocation of the registers to reduce SWAP count.'''
    anc = _allocs[_p]
    _allocs[_p] = _allocs[_q]
    _allocs[_q] = anc
    return _allocs

# ----------------------------------------------------------------
# Custom gates
# ----------------------------------------------------------------

def TR(_circ, inA, inB, outC):
    ''' Custom TR gate
    '''
    _circ.add(qibo.gates.X(inB))
    _circ.add(qibo.gates.TOFFOLI(inA, inB, outC))
    _circ.add(qibo.gates.X(inB))
    _circ.add(qibo.gates.CNOT(inA, inB))

def Peres(_circ, inA, inB, outC):
    ''' Custom Peres gate'''
    _circ.add(qibo.gates.TOFFOLI(inA, inB, outC))
    _circ.add(qibo.gates.CNOT(inA, inB))

def invCNOT(_circ, ctrl, target):
    '''Custom CNOT that fires when ctrl is |0>.'''
    _circ.add(qibo.gates.X(ctrl))
    _circ.add(qibo.gates.CNOT(ctrl, target))
    _circ.add(qibo.gates.X(ctrl))


# ----------------------------------------------------------------
# Conditional ADD/SUB (b -> b +/- a, selected by ctrl)
# ----------------------------------------------------------------

def add_sub_no_input_carry(_circ, ctrl, a, b):
    '''Ripple carry adder with no input carry bit.'''
    # Control operation
    for j in range(len(b)):
        _circ.add(qibo.gates.CNOT(ctrl, b[j]))

    # Block 1
    for j in range(len(b)-1):
        _circ.add(qibo.gates.CNOT(a[j+1], b[j+1]))

    # Block 2
    for j in range(len(a)-2):
        indx = len(a) - 1 - j
        _circ.add(qibo.gates.CNOT(a[indx-1], a[indx]))

    # Block 3
    for j in range(len(a)-1):
        _circ.add(qibo.gates.TOFFOLI(b[j], a[j], a[j+1]))

    # Block 4
    _circ.add(qibo.gates.CNOT(a[len(a)-1], b[len(b)-1]))

    # Block 5
    for j in range(len(a)-1):
        indx = len(a) - 1 - j
        Peres(_circ, a[indx-1], b[indx-1], a[indx])

    # Block 6
    for j in range(len(a)-2):
            _circ.add(qibo.gates.CNOT(a[j+1], a[j+2]))

    # Block 7
    for j in range(len(b)-1):
            _circ.add(qibo.gates.CNOT(a[j+1], b[j+1]))

    # Undo control operation
    for j in range(len(b)):
        _circ.add(qibo.gates.CNOT(ctrl, b[j]))

def _add_sub(_circ, ctrl, a, b):
    add_sub_no_input_carry(_circ, ctrl, a, b)



# ----------------------------------------------------------------
# Ctrl-ADD (b -> b + a if ctrl)
# ----------------------------------------------------------------
def ctrl_add(_circ, ctrl, a, b):
    # Block 1
    for i in range(len(b)-1):
        _circ.add(qibo.gates.CNOT(a[i+1], b[i+1]))

    # Block 2
    for i in range(len(a)-2):
        indx = len(a) - 1 - i
        _circ.add(qibo.gates.CNOT(a[indx-1], a[indx]))

    # Block 3
    for i in range(len(a)-1):
        _circ.add(qibo.gates.TOFFOLI(a[i], b[i], a[i+1]))

    # Block 4
    for i in range(len(b)-1):
        indx = len(b) - 1 - i
        _circ.add(qibo.gates.TOFFOLI(ctrl, a[indx], b[indx]))
        _circ.add(qibo.gates.TOFFOLI(b[indx-1] , a[indx-1], a[indx]))
    _circ.add(qibo.gates.TOFFOLI(ctrl, a[0], b[0]))

    # Block 5
    for i in range(len(b)-2):
        _circ.add(qibo.gates.CNOT(a[i+1], a[i+2]))

    # Block 6
    for i in range(len(b)-1):
            _circ.add(qibo.gates.CNOT(a[i+1], b[i+1]))


# ----------------------------------------------------------------
# Non-restoring SQRT steps
# ----------------------------------------------------------------
def initial_substraction(_circ, n, allocs): # Phase 1
    R, F, Z  = allocs

    # Overflow bug fix
    _circ.add(qibo.gates.CNOT(R[n-1], F[2]))
    _circ.add(qibo.gates.CNOT(R[n-2], F[2]))
    _circ.add(qibo.gates.TOFFOLI(R[n-1], R[n-2], F[2]))

    _circ.add(qibo.gates.X(R[n-2]))
    _circ.add(qibo.gates.CNOT(R[n-2], R[n-1]))

    invCNOT(_circ, F[2], F[1])

    # Step 6
    a = F[:4]
    b = R[n - 4:n]
    _add_sub(_circ, F[2], a, b)



def conditional_add_sub(_circ, n, it, allocs): # Phase 2 iteration
    R, F, Z  = allocs

    # Step 1
    invCNOT(_circ, F[2], F[1])

    # Step 3
    _circ.add(qibo.gates.CNOT(R[n-1], F[1]))

    # Step 5
    invCNOT(_circ, R[n-1], F[it+1])

    # Step 6
    for j in range(it+1,2,-1):
        #_circ.add(qibo.gates.SWAP(F[j], F[j-1]))
        F = swap_allocs(j, j-1, F)
    
    # Step 7
    a = F[:2 * it + 2]
    b = R[n - 2 * it - 2:n]
    _add_sub(_circ, F[2], a, b)


def remainder_restorarion(_circ, n, allocs): # Phase 3
    R, F, Z  = allocs



    # Step 1
    invCNOT(_circ, F[2], F[1])

    # Step 3
    invCNOT(_circ, R[n-1], Z)

    # Step 4
    invCNOT(_circ, R[n-1], F[n//2+1])

    # Step 5
    _circ.add(qibo.gates.X(Z))

    # Step 6
    ctrl_add(_circ, ctrl=Z, a=F, b=R)

    # Step 7
    _circ.add(qibo.gates.X(Z))

    # Step 8
    for j in range(n//2+1, 2, -1):
        #_circ.add(qibo.gates.SWAP(F[j], F[j-1]))
        F = swap_allocs(j, j-1, F)

    # Step 9
    _circ.add(qibo.gates.CNOT(F[2], Z))

