"""
Implementation of the randomized singular value decomposition.
"""
from numpy.random import randn
from numpy.linalg import matrix_power
from scipy.linalg import svd, qr
from scipy.linalg.interpolative import interp_decomp, reconstruct_interp_matrix


def rsvd(A, k, p=5):
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
    Q, _ = qr(Y, mode='economic', check_finite=False)

    # Get the projection B of A onto span(Q)
    B = Q.T @ A

    # Compute the SVD of B
    U_B, S, Vt = svd(B, full_matrices=False, lapack_driver="gesdd", check_finite=False)

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
    Q, _ = qr(Y, mode='reduced', check_finite=False)

    # Projection of A onto span(Q)
    B = Q.T @ A

    # SVD of B
    U_B, S, Vt = svd(B, full_matrices=False, lapack_driver="gesdd", check_finite=False)

    # Project U_B back to the original space
    U = Q @ U_B

    return U, S, Vt


def interpolative_decomp(A, k):
    # Compute the ID
    idx, proj = interp_decomp(A, k)

    # Index set J and selected columns AJ
    J = idx[:k]
    AJ = A[:, J]

    # Get X where X[:, J] = I_k 
    X = reconstruct_interp_matrix(idx, proj)

    return J, AJ, X


def rsvd3(A, k, p=5):
    m, n = A.shape

    # Obtain the approximate basis Q of A
    Omega = randn(n, min(k + p, m))
    Y = A @ Omega
    Q, _ = qr(Y, mode='reduced')

    # Compute the ID Q = QJ X and get the index set J
    J, QJ, X = interpolative_decomp(A, min(m, n))

    # Compute the QR factorization of AJ
    AJ = A[:, J]
    ...
