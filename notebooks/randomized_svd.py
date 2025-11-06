"""
Implementation of the randomized singular value decomposition.
"""
from numpy.random import randn
from numpy.linalg import qr, matrix_power
from scipy.linalg import svd


def rsvd(A, k, p = 3):
    """
    Given an m by n matrix A, a target rank k, and an oversampling parameter p,
    compute the approximate rank-k factorization `U S V^T` of A using the
    proto algorithm.
    """
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
    U_B, S, Vt = svd(B, full_matrices=False, lapack_driver="gesdd")

    # Project U_B back to the original space
    U = Q @ U_B

    # Return the k first vectors of the SVD
    return U[:, :k], S[:k], Vt[:k, :]


def rsvd2(A, k, q=2):
    """
    Given an m by n matrix A, a target rank 2k, and an exponent q,
    compute the approximate rank-2k factorization `U S V^T` of A using
    the two-stage algorithm.
    """
    m, n = A.shape

    # Random n by 2k Gaussian matrix
    Omega = randn(n, 2 * k)

    # Extract main "components" of A
    Y = matrix_power(A @ A.T, q) @ A @ Omega

    # Approximate orthogonal basis Q of A
    Q, _ = qr(Y, mode='reduced')

    # Projection of A onto span(Q)
    B = Q.T @ A

    # SVD of B
    U_B, S, Vt = svd(B, full_matrices=False, lapack_driver="gesdd")

    # Project U_B back to the original space
    U = Q @ U_B

    return U, S, Vt