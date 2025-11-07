import numpy as np


def exp_decay(n, C, alpha):
    """
    Generate an exponentially decaying sequence x_k = C * exp(-alpha * k).
    """
    k = np.arange(n)
    return C * np.exp(-alpha * k)


def poly_decay(n, C, alpha, q):
    """
    Generate a polynomial decaying sequence x_k = C / (1 + alpha * k)^q.
    """
    k = np.arange(n)
    return C / (1 + alpha * k) ** q


def random_orthogonal_matrix(n):
    """
    A random orthogonal matrix from an uniform distribution (distribution 
    given by Haar measure).
    """
    Omega = np.random.randn(n, n)
    Q, R = np.linalg.qr(Omega)
    return Q * np.sign(np.diag(R))


def random_test_matrix(m, n, singular_values):
    """
    Get a random test matrix with the given singular values.
    """
    S = np.zeros((m, n))
    ns = singular_values.size
    S[:ns, :ns] = np.diag(singular_values)
    U = random_orthogonal_matrix(m)
    V = random_orthogonal_matrix(n)
    return U @ S @ V.T


def N_random_test_matrices(N, m, n, singular_values):
    """
    Get N random test matrices of size (m, n) with the prescribed singular values.
    """
    A = np.zeros((N, m, n))
    for i in range(N):
        A[i, :, :] = random_test_matrix(m, n, singular_values)
    return A
