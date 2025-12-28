import numpy as np
from scipy.sparse.linalg import svds


def perform_svd(matrix, k=20):
    """
    Performs Singular Value Decomposition on the user-item matrix.

    Args:
        matrix (np.ndarray): User-Item interaction matrix.
        k (int): Number of latent factors to keep.

    Returns:
        prediction_matrix (np.ndarray): The reconstructed matrix with predicted ratings.
    """
    # Normalize by user mean ratings (only considering non-zero ratings)
    # Create a masked array where 0s are invalid
    masked_matrix = np.ma.masked_equal(matrix, 0)
    user_ratings_mean = masked_matrix.mean(axis=1).data
    matrix_demeaned = matrix - user_ratings_mean.reshape(-1, 1)

    # Perform SVD
    # U: User features, sigma: Singular values, Vt: Item features
    # Check if k is valid for the matrix size
    k = min(k, min(matrix.shape) - 1)

    U, sigma, Vt = svds(matrix_demeaned, k=k)

    # Convert sigma to diagonal matrix
    sigma = np.diag(sigma)

    # Reconstruct matrix
    all_user_predicted_ratings = np.dot(
        np.dot(U, sigma), Vt
    ) + user_ratings_mean.reshape(-1, 1)

    return all_user_predicted_ratings
