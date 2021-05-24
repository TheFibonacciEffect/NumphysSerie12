from numpy import (
    zeros,
    dot,
    diag,
    sqrt,
    ones,
    exp,
    complexfloating,
    linspace,
    conjugate,
    zeros_like,
)
from scipy.linalg import norm, eig, expm, inv, solve
import time


def construct_matrix(N, kind="minij"):
    def delta(i, j):
        return 1 if i == j else 0

    H = zeros((N, N), dtype=complexfloating)
    for i in range(N):
        for j in range(N):
            if kind == "sqrt":
                H[i, j] = 1 * sqrt((i + 1) ** 2 + (j + 1) ** 2) + (i + 1) * delta(i, j)
            elif kind == "dvr":
                if i != j:
                    t = i - j
                    H[i, j] = 2.0 / t ** 2
                    H[i, j] *= (-1) ** t
            elif kind == "minij":
                H[i, j] = float(1 + min(i, j))
            else:
                raise ValueError("Unknown matrix type: " + str(kind))

    if kind == "dvr":
        eps = 0.5
        a = -10.0
        b = 10.0
        h = (b - a) / N
        x = linspace(a, b, N)
        H *= (eps / h) ** 2
        H += diag(x ** 2)
        H *= dt / eps

    return H


def arnoldi(A, v0, k):
    """Arnoldi algorithm to compute the Krylov approximation.

    Input:
         A : The matrix `A` of shape `(N, N)` to approximate.
        v0 : The non-zero initial vector of length `N`.
         k : The number of Krylov steps to be performed.


    Ouput:
        V : is the large matrix of shape `(N, k+1)` containing the
             orthogonal vectors.

        H : is the small matrix of shape `(k, k)` containing the
            Krylov approximation of `A`.
    """
    r, c = A.shape
    V = zeros((r, k + 1), dtype=complexfloating)
    H = zeros((k + 1, k), dtype=complexfloating)

    # TODO: Implementieren Sie hier das Arnoldi Verfahren. #

    return V, H[:-1, :]


def lanczos(A, v0, k):
    """Lanczos algorithm to compute the Krylov approximation.

    Input:
         A : The matrix `A` of shape `(N, N)` to approximate.
        v0 : The non-zero initial vector of length `N`.
         k : The number of Krylov steps to be performed.


    Ouput:
        V : is the large matrix of shape `(N, k+1)` containing the
             orthogonal vectors.

        H : is the small matrix of shape `(k, k)` containing the
            Krylov approximation of `A`.
    """
    r, c = A.shape
    V = zeros((r, k + 1), dtype=complexfloating)
    H = zeros((k, k), dtype=complexfloating)

    # TODO: Implementieren Sie hier das Lanczos Verfahren. #

    return V, H


def integrate_krylov(A, v, dt, k, method):
    """Compute the solution of  `v' = -i A v`  via `k` steps of a Krylov method.

    Input:
            k : Number of Krylov (Arnoldi or Lanczos) steps
       method : Arnoldi or Lanczos function
    """
    vnew = zeros_like(v)

    ##########################################################################
    #                                                                        #
    # TODO: Loesen Sie die Differentialgleichung mit einem Krylov Verfahren. #
    #                                                                        #
    ##########################################################################

    return vnew


def integrate_diagonalize(A, v, dt):
    """Compute the solution of  `v' = -i A v`  via diagonalization of A."""
    vnew = zeros_like(v)

    # TODO: Loesen Sie die Differentialgleichung mittels Diagonalisierung. #

    return vnew


def integrate_exponential(A, v, dt):
    """Compute the solution of  `v' = -i A v`  via direct matrix exponential of A."""
    vnew = zeros_like(v)

    # TODO: Loesen Sie die Differentialgleichung mit dem direkten #
    #       Matrix Exponential 'expm'.                            #

    return vnew


k = 50
dt = 0.01
N = 2 ** 9

# TODO: Waehlen Sie die Matrix #
kind = "minij"
# kind = 'sqrt'
# kind = 'dvr'

A = construct_matrix(N, kind)

# initial value
v = ones(N, dtype=complex)
v = v / norm(v)

# Arnoldi
print("Exponential via Krylov Method: Arnoldi")
t0 = time.perf_counter()
yarnoldi = integrate_krylov(A, v, dt, k, arnoldi)
print(
    "  {:.3f} seconds for exponentiation with Arnoldi".format(time.perf_counter() - t0)
)

# Lanczos
print("Exponential via Krylov Method: Lanczos")
t0 = time.perf_counter()
ylanczos = integrate_krylov(A, v, dt, k, arnoldi)
print(
    "  {:.3f} seconds for exponentiation with Lanczos".format(time.perf_counter() - t0)
)

# Expm (Pade mehtod)
print("Exponential via Pade Approximation: expm")
t0 = time.perf_counter()
ymatexp = integrate_exponential(A, v, dt)
print(
    "  {:.3f} seconds for exponentiation with expm, i.e. Pade".format(
        time.perf_counter() - t0
    )
)

# Eig
print("Exponential via Eigendecomposition: eig")
ydiagon = integrate_diagonalize(A, v, dt)
print("  {:.3f} seconds for exponentiation with eig".format(time.perf_counter() - t0))

print("error = |method - expm| :")
print("Arnoldi-l2-error = {:.12f}".format(norm(yarnoldi - ymatexp)))
print("Lanczos-l2-error = {:.12f}".format(norm(ylanczos - ymatexp)))
print("error   |eAexpm - eig| : {:.12f}".format(norm(ydiagon - ymatexp)))
