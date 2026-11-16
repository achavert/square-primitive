''' Quantum squaring primitive (in-place, ancilla-free) by inverting the non-restoring square root

Register layout, with n=2*m (m = number of bits of the input x):

    R: n qubits -> holds x**2 at the end
    F: "root" register, its logical order is permuted while the sqrt runs. It owns no qubits of
       its own: F[2:m+2] are qubits of R. In particular F2 starts on R[1].

Total qubits: n  = 2m (no ancilla)

Why this works only as a squarer: bit 1 of a perfect square is always 0, so R[1] is a free |0> for the
whole sqrt except its last iteration, and it is used as the qubit that receives the first root bit.
The remainder of a perfect square is 0, so the remainder restoration is a controlled XOR instead of
a controlled addition. Therefore sqrt_circ is NOT a general square-root circuit: it is only correct
when its input is a perfect square, which is all that the inverse (the squarer) needs.

Differences with square_qubit_minimized.py (2m+1 qubits, C3X gates):
    - F2 is allocated on R[1] instead of on an extra qubit.
    - The conditional ADD/SUB is written as the single addition b -> b + 3 + 8V (see add_sub_no_ancilla),
      and its carry into position 3 is computed with F2 as a borrowed qubit: only X, CNOT and TOFFOLI.
    - Last iteration: R[1] is not available as b1. It is not needed (b1 = 0 on entry) and the carry
      becomes a single TOFFOLI.
    - Remainder restoration: ctrl_add is replaced by ctrl_clear_remainder (m-1 TOFFOLI).
'''

import qibo



def get_allocs(n):
    ''' Default initial allocations for the circuit:
    R: range(0, n)
    F: F0 = None, F1 = None, F2 = R[1]

    F0 = 1 and F1 = NOT F2 are virtual (no qubit). F2 sits on R[1], the bit of x**2 that is always 0.
    Custom allocs must keep this structure: F = [None, None, R[1]].
    '''
    R = list(range(0, n))
    F = [None, None, R[1]] # [F0, F1, F2]
    return R, F


def Square(m, allocs = None, input_swaps = False) -> qibo.Circuit:
    '''Builds the circuit that maps
            |x>|0>...|0>    ->  |x**2>
    on n = 2m qubits. Returns (circuit, (R,F)). The returned F is the final layout of the sqrt, which is
    where the input must be loaded:

        - input_swaps=False: load x directly on F[2:m+2] from the returned allocs.
        - input_swaps=True: load x on R[0:m], the circuits moves it to F with m extra SWAPs
          (m-1 for m=2, where the top bit is already in place).

    All qubits other than the ones holding x must start in |0>.
    The result x**2 is always on R[0:n], in little-endian configuration (least-significant bit on R[0]).
    '''
    if m < 2:
        raise ValueError("m must be at least 2")
    n = 2 * m
    nqubits = n

    R, F = get_allocs(n) if allocs is None else allocs
    if F[2] != R[1]:
        raise ValueError("F2 must be allocated on R[1]")
    sqrt_qc, (R, F) = sqrt_circ(n, (R, F))
    inv_sqrt_qc = sqrt_qc.invert()

    _qc = qibo.Circuit(nqubits)
    if input_swaps:
        # The order matters: R[1] is emptied (p=1) before it receives the top bit (p=m-1)
        for p in range(m):
            if R[p] != F[p+2]:
                _qc.add(qibo.gates.SWAP(R[p], F[p+2]))
    _qc.add(inv_sqrt_qc.on_qubits(*range(nqubits)))
    return _qc, (R, F)




def sqrt_circ(n, allocs) -> qibo.Circuit:
    ''' Non-restoring sqrt of an n-bit PERFECT SQUARE. Returns (circuit, (R,F)).

    The input 'allocs' is not modified, the returned F holds the final layout.
    '''
    R, F = allocs; F = list(F)
    nqubits = n
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

    n = 2 * m; nqubits = n
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

def Peres(_circ, inA, inB, outC):
    ''' Custom Peres gate'''
    _circ.add(qibo.gates.TOFFOLI(inA, inB, outC))
    _circ.add(qibo.gates.CNOT(inA, inB))


# ----------------------------------------------------------------
# Conditional ADD/SUB (b -> b +/- a, selected by ctrl)
# ----------------------------------------------------------------

def add_sub_no_ancilla(_circ, ctrl, a, b, last = False):
    '''Conditional ADD/SUB without F1, without ancilla and without C3X gates.

        - len(b) = N, len(a) = N-1, with a = [None, None, F2, F3, ...] and ctrl = a[2] = F2.
        - F[0]=a[0]=1 and F[1]=a[1]=NOT F[2] are virtual.

    With q = F2 and u = (..., F4, F3), the operation b +/- F (add if q=0, subtract if q=1), where
    F = 4*(2u+q) + 2*(NOT q) + 1, is the single addition

            b -> b + 3 + 8*V  (mod 2^N),        V = (q, u XOR q) read as a signed number

    so that:
        - the conditional complement acts on the addend qubits a[3:] instead of on the N qubits of b,
        - the three lowest addend bits are the constant 011 and need no qubit,
        - the ripple (positions 3 and above) only needs one carry from the low bits:
                carry = b2 AND (b1 OR b0)

    last = False: (b1, b0) are arbitrary. The carry is a product of three qubits. It is computed with
                  ctrl=F2 as a borrowed qubit: F2 holds F2 XOR [pair = 00] during the ripple, which never
                  reads it, and it is restored afterwards.
    last = True:  last iteration of the sqrt. b1 is R[1], the qubit that holds the top root bit, so it is
                  not available (b[1] is ignored). On a perfect square b1 = 0 on entry and its output
                  equals the output of b0, so it needs no qubit, and the carry is b2 AND b0.
    '''
    N = len(b)
    assert len(a) == N-1 and N >= 4
    q = ctrl
    assert q == a[2]

    def carry_in(t):
        ''' t -> t XOR carry '''
        if last:
            _circ.add(qibo.gates.TOFFOLI(b[2], b[0], t))
        else:
            _circ.add(qibo.gates.CNOT(b[2], t))
            _circ.add(qibo.gates.TOFFOLI(q, b[2], t))
            _circ.add(qibo.gates.TOFFOLI(b[0], b[1], q)) # q -> q XOR [pair = 00]
            _circ.add(qibo.gates.TOFFOLI(q, b[2], t))

    def carry_out(t):
        ''' Inverse of carry_in: clears the carry from t and restores q '''
        if last:
            _circ.add(qibo.gates.TOFFOLI(b[2], b[0], t))
        else:
            _circ.add(qibo.gates.TOFFOLI(q, b[2], t))
            _circ.add(qibo.gates.TOFFOLI(b[0], b[1], q)) # q restored
            _circ.add(qibo.gates.TOFFOLI(q, b[2], t))
            _circ.add(qibo.gates.CNOT(b[2], t))

    # Block 0: low pair (b1, b0) -> (b1, b0) - 1 mod 4. After it, b0 AND b1 = 1 only if the pair was 00.
    # In the last iteration it is delayed to the end, because the carry reads the original b0.
    if not last:
        _circ.add(qibo.gates.X(b[0]))
        _circ.add(qibo.gates.CNOT(b[0], b[1]))

    # Control operation: conditional complement of the addend and sign extension
    for j in range(3, N-1):
        _circ.add(qibo.gates.CNOT(q, a[j]))
    _circ.add(qibo.gates.CNOT(q, b[N-1]))

    if N == 4:
        # First iteration: there is no a[3], the carry is written directly on the top bit b[3]
        if last:
            carry_in(b[3])
            _circ.add(qibo.gates.CNOT(b[0], b[2]))
            _circ.add(qibo.gates.X(b[0]))
        else:
            carry_in(b[3])
            _circ.add(qibo.gates.CNOT(q, b[2]))
            _circ.add(qibo.gates.TOFFOLI(b[0], b[1], q)) # q restored
            _circ.add(qibo.gates.CNOT(q, b[2]))
            _circ.add(qibo.gates.X(b[2]))
        return

    # Block 1
    for j in range(3, N-1):
        _circ.add(qibo.gates.CNOT(a[j], b[j]))
    _circ.add(qibo.gates.CNOT(a[N-2], b[N-1]))

    # Block 2
    for j in range(N-2, 3, -1):
        _circ.add(qibo.gates.CNOT(a[j-1], a[j]))

    # Block 3
    carry_in(a[3])
    for j in range(3, N-2):
        _circ.add(qibo.gates.TOFFOLI(b[j], a[j], a[j+1]))
    _circ.add(qibo.gates.TOFFOLI(b[N-2], a[N-2], b[N-1]))

    # Block 4
    _circ.add(qibo.gates.CNOT(a[N-2], b[N-2]))
    for j in range(N-2, 3, -1):
        Peres(_circ, a[j-1], b[j-1], a[j])
    carry_out(a[3])

    # Block 5
    for j in range(4, N-1):
        _circ.add(qibo.gates.CNOT(a[j-1], a[j]))

    # Block 6
    for j in range(3, N-1):
        _circ.add(qibo.gates.CNOT(a[j], b[j]))

    # Sum bit of position 2 (b2 -> b2 XOR (b1 OR b0)), and delayed Block 0 in the last iteration
    if last:
        _circ.add(qibo.gates.CNOT(b[0], b[2]))
        _circ.add(qibo.gates.X(b[0]))
    else:
        _circ.add(qibo.gates.TOFFOLI(b[0], b[1], b[2]))
        _circ.add(qibo.gates.X(b[2]))

    # Undo control operation
    for j in range(3, N-1):
        _circ.add(qibo.gates.CNOT(q, a[j]))


def _add_sub(_circ, ctrl, a, b, last = False):
    add_sub_no_ancilla(_circ, ctrl, a, b, last)



# ----------------------------------------------------------------
# Remainder restoration (replaces Ctrl-ADD)
# ----------------------------------------------------------------
def ctrl_clear_remainder(_circ, ctrl, a, b):
    ''' Clears the remainder register b = R[0:m+1] of a perfect square (b[1] has no qubit and is ignored).

    The remainder of a perfect square is 0, so before the restoration b holds
        - 0            if ctrl = 0
        - -(2y+1)      if ctrl = 1, that is  b0 = 1, b1 = 1, bk = NOT a[k] for k >= 2
    and the controlled addition of 2y+1 reduces to a controlled XOR with that pattern.
    '''
    _circ.add(qibo.gates.CNOT(ctrl, b[0]))
    for i in range(2, len(b)):
        _circ.add(qibo.gates.CNOT(ctrl, b[i]))
        _circ.add(qibo.gates.TOFFOLI(ctrl, a[i], b[i]))


# ----------------------------------------------------------------
# Non-restoring SQRT steps
# ----------------------------------------------------------------
def initial_substraction(_circ, n, allocs): # Phase 1
    R, F = allocs

    # Corrected control computation. F[2] is R[1], which is |0> for a perfect square
    _circ.add(qibo.gates.CNOT(R[n-1], F[2]))
    _circ.add(qibo.gates.CNOT(R[n-2], F[2]))
    _circ.add(qibo.gates.TOFFOLI(R[n-1], R[n-2], F[2]))

    # Initial subtraction computation
    _circ.add(qibo.gates.X(R[n-2]))
    _circ.add(qibo.gates.CNOT(R[n-2], R[n-1]))

    # Step 4 (j=1 conditional ADD/SUB). For m=2 this is already the last iteration
    a = F[:3] # a = [None, None, F2]
    b = R[n - 4:n]
    _add_sub(_circ, F[2], a, b, last = (n//2 == 2))




def conditional_add_sub(_circ, n, it, allocs): # Phase 2 iteration
    R, F  = allocs

    sign = R[n - it + 1] # Sign will be stored at the top bit in each iteration

    _circ.add(qibo.gates.X(sign)) # Step 3: prepare actual root bit, opposite to sign

    F.append(sign) # Sign qubit becomes root bit

    for j in range(it+1,2,-1): # Step 4: qubit reallocation, replacing _circ.add(qibo.gates.SWAP(F[j], F[j-1]))
        F = swap_allocs(j, j-1, F)

    # Step 5: perform conditional ADD/SUB over the restricted qubit window fixed by F (it+2) and R (it+3) bounds
    N = it + 3
    a = F[: N - 1]
    b = R[n - 2 * it - 2: n - 2 * it - 2+ N]
    _add_sub(_circ, F[2], a, b, last = (it == n//2 - 1))


def remainder_restorarion(_circ, n, allocs): # Phase 3
    R, F  = allocs

    h = n // 2
    sign = R[h+1] # Sign qubit

    # Step 2: clear the remainder on the narrowed qubits R[0:h+1] (R[1] excluded, it holds a root bit)
    a = F[:h+1]
    b = R[:h+1]
    ctrl = sign
    ctrl_clear_remainder(_circ, ctrl, a, b)

    _circ.add(qibo.gates.X(sign)) #Step 3: prepare actual root bit, opposite to sign
    F.append(sign)

    # Step 5: qubit reallocation, replacing _circ.add(qibo.gates.SWAP(F[j], F[j-1]))
    for j in range(n//2+1, 2, -1):
        F = swap_allocs(j, j-1, F)
