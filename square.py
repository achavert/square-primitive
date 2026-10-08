import numpy as np
import qibo
from qibo import Circuit, gates


def Square(m, method = 'input_carry', **kargs) -> qibo.Circuit:
    n = 2*m
    R, F, Z = get_allocs(n)
    total_qubits = 2*n + 1
    _qc = qibo.Circuit(total_qubits)
    
    for p in range(m): # Reallocs
        _qc.add(gates.SWAP(R[p], F[p+2]))

    _square_circuit = square_circ(m, method, **kargs)
    _qc.add(_square_circuit.on_qubits(*range(total_qubits)))
    _qc.add(gates.X(F[0]))
    return _qc


def square_circ(m, method = 'input_carry', **kargs) -> qibo.Circuit:
    n = 2*m
    R, F, Z = get_allocs(n)
    total_qubits = 2*n + 1
    _qc_square = qibo.Circuit(total_qubits)
    _qc_square.add(gates.X(F[0]))
    _sqrt_circ = sqrt_circ(n, method, **kargs)
    inv_sqrt = _sqrt_circ.invert()
    _qc_square.add(inv_sqrt.on_qubits(*range(total_qubits)))
    return _qc_square


def sqrt_circ(n, method = 'input_carry', **kargs) -> qibo.Circuit:
    _qc_sqrt = Circuit(2*n + 1)
    initial_substraction(_qc_sqrt, n, method, **kargs)
    for it in range(2, n//2):
        conditional_add_sub(_qc_sqrt, n, it, method, **kargs)
    remainder_restorarion(_qc_sqrt, n)
    return _qc_sqrt


def initialize_square(x, m) -> qibo.Circuit:
    n = 2*m
    total_qubits = 2*n + 1
    init_circ = qibo.Circuit(total_qubits)
    R, F, Z = get_allocs(n)
    square_bits = R[0:n//2] 
    if not 0 <= x < 2**m:
        raise ValueError(f"x={x} does not fit in {m} bits")
    for i in range(m):
        if (x >> i) & 1:
            init_circ.add(gates.X(square_bits[i]))
    
    return init_circ
# ----------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------
def get_allocs(n):
    R = list(range(0, n))             
    F = list(range(n, 2*n))        
    Z = 2*n    
    return R, F, Z    

# ----------------------------------------------------------------
# Custom gates
# ----------------------------------------------------------------

def TR(_circ, inA, inB, outC):
    _circ.add(gates.X(inB))
    _circ.add(gates.TOFFOLI(inA, inB, outC))
    _circ.add(gates.X(inB))
    _circ.add(gates.CNOT(inA, inB))

def Peres(_circ, inA, inB, outC):
    _circ.add(gates.TOFFOLI(inA, inB, outC))
    _circ.add(gates.CNOT(inA, inB))

def invCNOT(_circ, ctrl, target):
    _circ.add(gates.X(ctrl))
    _circ.add(gates.CNOT(ctrl, target))
    _circ.add(gates.X(ctrl))


# ----------------------------------------------------------------
# Conditional ADD/SUB
# ----------------------------------------------------------------

def add_sub_no_input_carry(_circ, ctrl, a, b): # Approach 2: without input carry
    # Control operation
    for j in range(len(b)):
        _circ.add(gates.CNOT(ctrl, b[j]))

    # Block 1
    for j in range(len(b)-1):
        _circ.add(gates.CNOT(a[j+1], b[j+1]))

    # Block 2
    for j in range(len(a)-2):
        indx = len(a) - 1 - j
        _circ.add(gates.CNOT(a[indx-1], a[indx]))

    # Block 3
    for j in range(len(a)-1):
        _circ.add(gates.TOFFOLI(b[j], a[j], a[j+1]))

    # Block 4
    _circ.add(gates.CNOT(a[len(a)-1], b[len(b)-1]))

    # Block 5
    for j in range(len(a)-1):
        indx = len(a) - 1 - j
        Peres(_circ, a[indx-1], b[indx-1], a[indx])

    # Block 6
    for j in range(len(a)-2):
            _circ.add(gates.CNOT(a[j+1], a[j+2]))

    # Block 7
    for j in range(len(b)-1):
            _circ.add(gates.CNOT(a[j+1], b[j+1]))

    # Undo control operation
    for j in range(len(b)):
        _circ.add(gates.CNOT(ctrl, b[j]))

def add_sub_input_carry(_circ, ctrl, a, b): # Approach 3: with input carry
    # Control operation with the carry bit
    for j in range(len(a)):
        _circ.add(gates.CNOT(ctrl, a[j]))

    # Block 1
    for j in range(len(b)):
        _circ.add(gates.CNOT(a[j], b[j]))

    # Block 2
    _circ.add(gates.CNOT(a[0], ctrl))
    for j in range(len(a)-1):
        _circ.add(gates.CNOT(a[j+1], a[j]))

    # Block 3
    _circ.add(gates.TOFFOLI(ctrl, b[0], a[0]))
    for j in range(len(a)-2):
        _circ.add(gates.TOFFOLI(a[j], b[j+1], a[j+1]))

    # Block 4
    _circ.add(gates.CNOT(a[-2], b[-1]))

    # Block 5
    for j in range(len(b)-1):
        _circ.add(gates.X(b[j]))

    # Block 6
    for j in range(len(a)-2):
        indx = len(a) - 2 - j
        TR(_circ, a[indx-1], b[indx], a[indx])
    TR(_circ, ctrl, b[0], a[0])

    # Block 7
    for j in range(len(b)-1):
        _circ.add(gates.X(b[j]))

    # Block 8
    for j in range(len(a)-1):
        indx = len(a) - 1 - j
        _circ.add(gates.CNOT(a[indx], a[indx-1]))
    _circ.add(gates.CNOT(a[0], ctrl))

    # Block 9
    for j in range(len(b)):
        _circ.add(gates.CNOT(a[j], b[j]))

    # Undo control operation with the carry bit
    for j in range(len(a)):
        _circ.add(gates.CNOT(ctrl, a[j]))


def add_sub_aqft(_circ, ctrl, a, b, **kargs): # Approach 2: without input carry
    delta = kargs.get('delta', len(b))
    parallel = kargs.get('parallel', False)

    # Control operation
    for j in range(len(b)):
        _circ.add(gates.CNOT(ctrl, b[j]))

    aqft(_circ, b, delta)

    if parallel:
        for _l in range(min(len(a), delta)):                  
            for _i in range(min(len(b), len(a) - _l)):        
                _ctrl = a[len(a)-1 - (_i + _l)]               
                _targ = b[len(b)-1 - _i]                      
                theta = np.pi / 2**_l
                _circ.add(gates.CU1(_ctrl, _targ, theta))

    else:
        for _l in range(len(b)):
            _targ = b[len(b)-1 - _l]
            for _k in range(0, min(len(a) - _l, delta)):
                _ctrl = a[len(a)-1 - _l - _k] 
                theta = np.pi / 2**_k
                _circ.add(gates.CU1(_ctrl, _targ, theta))

    iaqft(_circ, b, delta)

    # Undo control operation
    for j in range(len(b)):
            _circ.add(gates.CNOT(ctrl, b[j]))


def aqft(_circ, b, delta):
    nb = len(b)
    for j in range(nb-1, -1, -1):
        _circ.add(gates.H(b[j]))
        for g in range(j-1, max(j - delta, 0) -1, -1):
            _circ.add(gates.CU1(b[g], b[j], np.pi/2**(j-g)))

def iaqft(_circ, b, delta):
    nb = len(b)
    for j in range(nb):
        for g in range(max(j - delta, 0), j):
            _circ.add(gates.CU1(b[g], b[j], -np.pi/2**(j-g)))
        _circ.add(gates.H(b[j]))





# ----------------------------------------------------------------
# Ctrl-ADD
# ----------------------------------------------------------------
def ctrl_add(_circ, ctrl, a, b):
    # Block 1
    for i in range(len(b)-1):
        _circ.add(gates.CNOT(a[i+1], b[i+1]))

    # Block 2
    for i in range(len(a)-2):
        indx = len(a) - 1 - i
        _circ.add(gates.CNOT(a[indx-1], a[indx]))

    # Block 3
    for i in range(len(a)-1):
        _circ.add(gates.TOFFOLI(a[i], b[i], a[i+1]))

    # Block 4
    for i in range(len(b)-1):
        indx = len(b) - 1 - i
        _circ.add(gates.TOFFOLI(ctrl, a[indx], b[indx]))
        _circ.add(gates.TOFFOLI(b[indx-1] , a[indx-1], a[indx]))
    _circ.add(gates.TOFFOLI(ctrl, a[0], b[0]))

    # Block 5
    for i in range(len(b)-2):
        _circ.add(gates.CNOT(a[i+1], a[i+2]))

    # Block 6
    for i in range(len(b)-1):
            _circ.add(gates.CNOT(a[i+1], b[i+1]))


# ----------------------------------------------------------------
# Non-restoring SQRT steps
# ----------------------------------------------------------------
def initial_substraction(_circ, n, method = 'input_carry', **kargs): # Phase 1
    R, F, Z  = get_allocs(n)

    ''' Overflow bugged circuit
    # Step 1
    _circ.add(gates.X(R[n-2]))

    # Step 2
    _circ.add(gates.CNOT(R[n-2], R[n-1]))

    # Step 3
    _circ.add(gates.CNOT(R[n-1], F[1]))

    # Step 4
    invCNOT(_circ, R[n-1], Z)

    # Step 5
    invCNOT(_circ, R[n-1], F[2])
    '''

    # Overflow bug fix

    _circ.add(gates.CNOT(R[n-1], Z))
    _circ.add(gates.CNOT(R[n-2], Z))
    _circ.add(gates.TOFFOLI(R[n-1], R[n-2], Z))


    _circ.add(gates.X(R[n-2]))
    _circ.add(gates.CNOT(R[n-2], R[n-1]))

    _circ.add(gates.CNOT(Z, F[2]))
    invCNOT(_circ, Z, F[1])


    # Step 6
    a = [F[0], F[1], F[2], F[3]]
    b = [R[n - 4], R[n - 3], R[n - 2], R[n - 1]]
    if method == 'input_carry':
        add_sub_input_carry(_circ, Z, a, b)
    elif method == 'no_input_carry':
        add_sub_no_input_carry(_circ, Z, a, b)
    elif method == 'aqft':
        add_sub_aqft(_circ, Z, a, b, **kargs)
    else:
        raise ValueError('ADD/SUB method not valid.')



def conditional_add_sub(_circ, n, it, method = 'input_carry', **kargs): # Phase 2 iteration
    R, F, Z  = get_allocs(n)

    # Step 1
    invCNOT(_circ, Z, F[1])

    # Step 2
    _circ.add(gates.CNOT(F[2], Z))

    # Step 3
    _circ.add(gates.CNOT(R[n-1], F[1]))

    # Step 4
    invCNOT(_circ, R[n-1], Z)

    # Step 5
    invCNOT(_circ, R[n-1], F[it+1])

    # Step 6
    for j in range(it+1,2,-1):
        _circ.add(gates.SWAP(F[j], F[j-1]))

    # Step 7
    a = [F[j] for j in range(0, 2*it +2)]
    b = [R[j] for j in range(n-2*it-2, n)]
    if method == 'input_carry':
        add_sub_input_carry(_circ, Z, a, b)
    elif method == 'no_input_carry':
        add_sub_no_input_carry(_circ, Z, a, b)
    elif method == 'aqft':
            add_sub_aqft(_circ, Z, a, b, **kargs)
    else:
        raise ValueError('ADD/SUB method not valid.')


def remainder_restorarion(_circ, n): # Phase 3
    R, F, Z  = get_allocs(n)

    # Step 1
    invCNOT(_circ, Z, F[1])

    # Step 2
    _circ.add(gates.CNOT(F[2], Z))

    # Step 3
    invCNOT(_circ, R[n-1], Z)

    # Step 4
    invCNOT(_circ, R[n-1], F[n//2+1])

    # Step 5
    _circ.add(gates.X(Z))

    # Step 6
    ctrl_add(_circ, ctrl=Z, a=F, b=R)

    # Step 7
    _circ.add(gates.X(Z))

    # Step 8
    for j in range(n//2+1, 2, -1):
        _circ.add(gates.SWAP(F[j], F[j-1]))

    # Step 9
    _circ.add(gates.CNOT(F[2], Z))


