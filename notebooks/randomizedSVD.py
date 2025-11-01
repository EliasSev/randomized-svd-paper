"""
Implementation of the randomized singular value decomposition.
"""
from numpy.random import randn
from numpy.linalg import qr, svd


def rsvd(A, k, p = 3):
    m, n = A.shape

    # Random matrix from a standard Gaussian distribution
    Omega = randn(n, min(k + p, m))

    # Obtain the n x (k + p) matrix Y = (A)(Omega)
    Y = A @ Omega

    # Get the orthonormal basis Q of Y
    Q, _ = qr(Y, mode='reduced')

    # Get the projection B of A onto span(Q)
    B = Q.T @ A

    # Compute the SVD of B
    U_B, S, Vt = svd(B, full_matrices=False)

    # Project U_B back to the original space
    U = Q @ U_B

    # Return the k first vectors of the SVD
    return U[:, :k], S[:k], Vt[:k, :]
