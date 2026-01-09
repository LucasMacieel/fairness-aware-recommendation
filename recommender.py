import numpy as np
from surprise import SVD


def train_svd_surprise(trainset, random_state=42):
    """
    Train an SVD model using scikit-surprise.

    Args:
        trainset: A Surprise Trainset object (from build_full_trainset or train_test_split)
        n_factors (int): Number of latent factors for SVD
        random_state (int): Random seed for reproducibility

    Returns:
        SVD: Trained SVD model
    """
    algo = SVD(random_state=random_state, biased=False)
    algo.fit(trainset)
    return algo


def get_predictions_matrix(algo, user_ids, item_ids, trainset):
    """
    Generate a full prediction matrix for all user-item pairs.

    Args:
        algo: Trained Surprise SVD model
        user_ids: List of original user IDs (in matrix row order)
        item_ids: List of original item IDs (in matrix column order)
        trainset: The Trainset used for training (needed for inner ID mapping)

    Returns:
        np.ndarray: Prediction matrix of shape (num_users, num_items)
    """
    num_users = len(user_ids)
    num_items = len(item_ids)
    prediction_matrix = np.zeros((num_users, num_items))

    for u_idx, uid in enumerate(user_ids):
        for i_idx, iid in enumerate(item_ids):
            # Surprise returns prediction with .est attribute
            pred = algo.predict(uid, iid)
            prediction_matrix[u_idx, i_idx] = pred.est

    return prediction_matrix
