
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
    # Normalize by user mean ratings
    user_ratings_mean = np.mean(matrix, axis=1)
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
