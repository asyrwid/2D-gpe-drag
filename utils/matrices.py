import numpy as np
import scipy as sp
from scipy import sparse
from numpy.linalg import inv

from utils.structures import TrapParameters


def laplacian2D_pbc(N: int):
    diag = np.ones([N * N])
    mat = sparse.spdiags([diag, diag, -2*diag, diag, diag],
                         [-N+1, -1, 0, 1, N-1], N, N)
    Identity = sp.eye(N)
    lap = sparse.kron(Identity, mat, format='csr') + sparse.kron(mat, Identity)
    return lap  # return sparse array


def inv_matrix1(
    laplacian2d,
    dx: float,
    dt: float,
    trap_parameters: TrapParameters,
) -> np.array:
    # inverse of O1 matrix generation
    wx = trap_parameters.wx
    wy = trap_parameters.wy
    xshift = trap_parameters.xshift
    yshift = trap_parameters.yshift
    O1 = laplacian2d.toarray()
    O1 = np.array(O1, dtype=np.complex128)/(4.*dx*dx)  # dy = dx
    N = int(len(O1)**0.5)
    n0 = (N - 1.0)/2.
    s = 0
    for i in range(0, N):
        for j in range(0, N):
            x2 = (dx*(i - n0) - xshift)**2
            y2 = (dx*(j - n0) - yshift)**2
            add = -1/4. * ((wx**2)*x2 + (wy**2)*y2) + 1j*1./dt
            O1[s, s] += add
            s = s + 1
    return inv(O1)


def matrix2(
    laplacian2d,
    dx: float,
    dt: float,
    trap_parameters: TrapParameters,
) -> np.array:
    # O2 matrix generation
    wx = trap_parameters.wx
    wy = trap_parameters.wy
    xshift = trap_parameters.xshift
    yshift = trap_parameters.yshift
    O2 = -laplacian2d.toarray()
    O2 = np.array(O2, dtype=np.complex128)/(4.*dx*dx)
    N = int(len(O2)**0.5)
    n0 = (N - 1.0)/2.
    s = 0
    for i in range(0, N):
        for j in range(0, N):
            x2 = (dx*(i - n0) - xshift)**2
            y2 = (dx*(j - n0) - yshift)**2
            add = 1/4. * ((wx**2)*x2 + (wy**2)*y2) + 1j*1./dt
            O2[s, s] += add
            s = s + 1
    return O2
