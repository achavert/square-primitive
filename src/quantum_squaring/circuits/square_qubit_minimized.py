''' Quantum squaring primitive (in-place) by inverting the non-restoring square root

Register layout, with n=2*m (m = number of bits of the input x):

    R: n qubits -> holds x**2 at the end
    F: n qubits -> "root" register, its logical order is permuted while the sqrt runs

Total qubits: n+2  = 2m+2
'''

import qibo



def get_allocs(n):
    ''' Default initial allocations for the circuit:
    R: range(0, n)
    F: F0 = None, F1 = n, F2 = n+1
    '''
    R = list(range(0, n))             
    F = [None, None, n] # [F0, F1, F2]         
    return R, F   


def Square(m, allocs = None, input_swaps = False) -> qibo.Circuit:
    '''Builds the circuit that maps 
            |x>|0>...|0>    ->  |x**2>|0>...|0>
    Returns (circuit, (R,F)). The returned F is the final layout of the sqrt, which is where the input must be loaded:

        - input_swaps=False: load x directly on F[2:m+2] from the returned allocs.
        - input_swaps=True: load x on R[0:m], the circuits moves it to F with m extra SWAPs

    The result x**2 is always on R[0:n], in little-endian configuration (least-significant bit on R[0]).     
    '''
    n = 2 * m
    nqubits = n + 1

    R, F = get_allocs(n) if allocs is None else allocs
    sqrt_qc, (R, F) = sqrt_circ(n, (R, F))
    inv_sqrt_qc = sqrt_qc.invert()

    _qc = qibo.Circuit(nqubits)
    if input_swaps:
        for p in range(m):
            _qc.add(qibo.gates.SWAP(R[p], F[p+2]))
    _qc.add(inv_sqrt_qc.on_qubits(*range(nqubits)))
    return _qc, (R, F)




def sqrt_circ(n, allocs) -> qibo.Circuit:
    ''' Non-restoring sqrt of an n-bit number. Returns (circuit, (R,F)).

    The input 'allocs' is not modified, the returned F holds the final layout.
    '''
    R, F = allocs; F = list(F)
    nqubits = n + 1
    _qc_sqrt = qibo.Circuit(nqubits)

    initial_substraction(_qc_sqrt, n, (R, F))
    for it in range(2, n//2):
        conditional_add_sub(_qc_sqrt, n, it, (R, F))
    remainder_restorarion(_qc_sqrt, n, (R, F))
    return _qc_sqrt, (R, F)


def initialize_square(x, m, allocs = None, direct = False) -> qibo.Circuit:
    ''' Circuit that loads the m-bit integer x (little-endian):
            - direct=False: onto R[0:m] (use with input_swaps=True).
            - direct=True:  onto F[2:m+2] of the final layout, allocs must be the ones returned by Square
    '''
    if not 0 <= x < 2**m:
        raise ValueError(f"x={x} does not fit in {m} bits")
    if direct and allocs is None:
        raise ValueError("direct=True needs the allocs returned by Square")

    n = 2 * m; nqubits = n + 1
    R, F = get_allocs(n) if allocs is None else allocs
    register = F[2:m+2] if direct else R[:m]

    init_circ = qibo.Circuit(nqubits)
    for i, q in enumerate(register):
        if (x >> i) & 1:
            init_circ.add(qibo.gates.X(q))
    return init_circ

# ----------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------

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
from qibo import gates



def add_sub_no_input_carry(_circ, ctrl, a, b):
    '''Optimized ripple carry adder with no input carry bit (n+1 / 2m+1 version).
        - len(b) = N, len(a) = N-1 (F[N-1] is always zero by theoretical bounds).
        - F[0]=a[0]=1 and F[1]=NOT F[2] are virtualized (a[1] is not used / None).
        - The qubit q = a[2] encodes F2 and its negation handles F1.
        Requires N >= 4.
    '''
    N = len(b)
    assert len(a) == N-1 and N >= 4
    q = a[2]
 
    def a2_toffoli(t):                 
        _circ.add(gates.CNOT(b[2], t))
        # Puerta C3X (Toffoli con 3 controles: b[1], b[2], q) para el caso n+1
        gate = gates.X(t).controlled_by(b[1], b[2], q)
        _circ.add(gate)
 
    def a2_cnot_b2():                 
        _circ.add(gates.X(b[2]))
        _circ.add(gates.TOFFOLI(b[1], q, b[2]))
 
    # Control operation
    for j in range(N):
        _circ.add(gates.CNOT(ctrl, b[j]))
 
    # Block 1
    _circ.add(gates.X(b[1]))
    _circ.add(gates.CNOT(q, b[1]))
    _circ.add(gates.CNOT(q, b[2]))
    for j in range(len(a)-3):
        _circ.add(gates.CNOT(a[j+3], b[j+3]))
    _circ.add(gates.CNOT(a[N-2], b[N-1]))

    # Block 2
    for j in range(len(a)-3):
        indx = len(a) - 1 - j
        _circ.add(gates.CNOT(a[indx-1], a[indx]))
    _circ.add(gates.X(q))

    # Block 3
    _circ.add(gates.CNOT(b[0], q))
    if N == 4:
        a2_toffoli(b[3])
    else:
        a2_toffoli(a[3])
        for j in range(len(a)-4):
            _circ.add(gates.TOFFOLI(b[j+3], a[j+3], a[j+4]))
        _circ.add(gates.TOFFOLI(b[N-2], a[N-2], b[N-1]))

    # Block 4
    if N == 4:
        a2_cnot_b2()
    else:
        _circ.add(gates.CNOT(a[N-2], b[N-2]))
        for j in range(len(a)-4):
            indx = len(a) - 1 - j
            # Asumiendo que Peres está definida en el contexto del módulo
            Peres(_circ, a[indx-1], b[indx-1], a[indx])
        a2_toffoli(a[3])
        a2_cnot_b2()
    _circ.add(gates.CNOT(q, b[1]))
    _circ.add(gates.CNOT(b[0], q))
    _circ.add(gates.X(b[0]))
    _circ.add(gates.X(q))

    # Block 5
    for j in range(len(a)-3):
        _circ.add(gates.CNOT(a[j+2], a[j+3]))

    # Block 6 (Inversión exacta y simétrica del Bloque 1 para n+1)
    for j in reversed(range(len(a)-3)):
        _circ.add(gates.CNOT(a[j+3], b[j+3]))
    _circ.add(gates.CNOT(q, b[2]))
    _circ.add(gates.CNOT(q, b[1]))
    _circ.add(gates.X(b[1]))

    # Undo control operation
    for j in range(N):
        _circ.add(gates.CNOT(ctrl, b[j]))


def _add_sub(_circ, ctrl, a, b):
    add_sub_no_input_carry(_circ, ctrl, a, b)



# ----------------------------------------------------------------
# Ctrl-ADD (b -> b + a if ctrl)
# ----------------------------------------------------------------
def ctrl_add(_circ, ctrl, a, b):
    # Block 1
    for i in range(len(b)-2): # Removed first a1=0
        _circ.add(qibo.gates.CNOT(a[i+2], b[i+2]))

    # Block 2
    for i in range(len(a)-3) : # Removed last a1=0
        indx = len(a) - 1 - i
        _circ.add(qibo.gates.CNOT(a[indx-1], a[indx]))

    # Block 3
    #_circ.add(qibo.gates.CNOT(b[0], a[1])) # Changed Toffoli gate with control in F[0]=1 for a CNOT
    _circ.add(qibo.gates.TOFFOLI(b[0], b[1], a[2]))
    for i in range(len(a)-3):
        _circ.add(qibo.gates.TOFFOLI(a[i+2], b[i+2], a[i+3])) # Changed Toffoli gate with control in F[0]=1 for a CNOT

    for i in range(len(b)-3):
        indx = len(b) - 1 - i
        _circ.add(qibo.gates.TOFFOLI(ctrl, a[indx], b[indx]))
        _circ.add(qibo.gates.TOFFOLI(b[indx-1] , a[indx-1], a[indx]))
    _circ.add(qibo.gates.TOFFOLI(ctrl, a[2], b[2]))
    _circ.add(qibo.gates.TOFFOLI(b[0] , b[1], a[2]))
    
    _circ.add(qibo.gates.TOFFOLI(ctrl, b[0], b[1]))
    _circ.add(qibo.gates.CNOT(ctrl, b[0])) # Changed Tofolli with control on a0=F[0]=1 for a CNOT

    # Block 5
    for i in range(len(b)-3):
        _circ.add(qibo.gates.CNOT(a[i+2], a[i+3]))

    # Block 6
    for i in range(len(b)-2):
            _circ.add(qibo.gates.CNOT(a[i+2], b[i+2]))
# ----------------------------------------------------------------
# Non-restoring SQRT steps
# ----------------------------------------------------------------
def initial_substraction(_circ, n, allocs): # Phase 1
    R, F = allocs

    # Corrected control computation
    _circ.add(qibo.gates.CNOT(R[n-1], F[2]))
    _circ.add(qibo.gates.CNOT(R[n-2], F[2]))
    _circ.add(qibo.gates.TOFFOLI(R[n-1], R[n-2], F[2]))

    # Initial subtraction computation
    _circ.add(qibo.gates.X(R[n-2]))
    _circ.add(qibo.gates.CNOT(R[n-2], R[n-1]))

    # Bit inversion for F1 (F2 and F1 are always opposites)
    #invCNOT(_circ, F[2], F[1])

    # Step 4 (j=1 conditional ADD/SUB)
    a = F[:3] # F3 is always 0, so it can be removed => a = [None, F1, F2]
    b = R[n - 4:n]
    _add_sub(_circ, F[2], a, b)




def conditional_add_sub(_circ, n, it, allocs): # Phase 2 iteration
    R, F  = allocs

    sign = R[n - it + 1] # Sign will be stored at the top bit in each iteration
    
    #invCNOT(_circ, F[2], F[1]) # Step 1: clear F1

    #_circ.add(qibo.gates.CNOT(sign, F[1])) # Step 2: copy value from sign to F1

    _circ.add(qibo.gates.X(sign)) # Step 3: prepare actual root bit, opposite to sign

    F.append(sign) # Sign qubit becomes root bit

    for j in range(it+1,2,-1): # Step 4: qubit reallocation, replacing _circ.add(qibo.gates.SWAP(F[j], F[j-1]))
        F = swap_allocs(j, j-1, F) 
    
    # Step 5: perform conditional ADD/SUB over the restricted qubit window fixed by F (it+2) and R (it+3) bounds
    N = it + 3
    a = F[: N - 1]
    b = R[n - 2 * it - 2: n - 2 * it - 2+ N]
    _add_sub(_circ, F[2], a, b) 


def remainder_restorarion(_circ, n, allocs): # Phase 3
    R, F  = allocs

    h = n // 2
    sign = R[h+1] # Sign qubit

    #invCNOT(_circ, F[2], F[1]) # Step 1: Clear F1

    # Step 2: CTRL-ADD on the narrowed qubits from n-1 to h+1
    a = F[:h+1]
    b = R[:h+1]
    ctrl = sign
    ctrl_add(_circ, ctrl, a, b)

    _circ.add(qibo.gates.X(sign)) #Step 3: prepare actual root bit, opposite to sign
    F.append(sign)

    # Step 5: qubit reallocation, replacing _circ.add(qibo.gates.SWAP(F[j], F[j-1]))
    for j in range(n//2+1, 2, -1):
        F = swap_allocs(j, j-1, F)



