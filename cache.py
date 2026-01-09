import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

# Default cache directory in project root
DEFAULT_CACHE_DIR: Path = Path(__file__).parent / "cache"


def get_cache_dir() -> Path:
    """Get the cache directory path, creating it if it doesn't exist."""
    cache_dir = DEFAULT_CACHE_DIR
    cache_dir.mkdir(exist_ok=True)
    return cache_dir


def _sanitize_dataset_name(dataset_name: str) -> str:
    """Sanitize dataset name for filesystem usage."""
    return dataset_name.lower().replace(" ", "_").replace("-", "_")


def get_cache_path(dataset_name: str, matrix_type: str) -> Path:
    """
    Generate cache file path based on dataset and matrix type.

    Args:
        dataset_name: Name of the dataset (e.g., 'MovieLens 100k')
        matrix_type: Type of matrix (e.g., 'train', 'val', 'test', 'prediction')

    Returns:
        Path: Full path to the cache file
    """
    safe_name = _sanitize_dataset_name(dataset_name)
    filename = f"{safe_name}_{matrix_type}.npy"
    return get_cache_dir() / filename


def save_matrix(matrix: NDArray[np.floating], cache_path: Path) -> None:
    """
    Save numpy array to disk.

    Args:
        matrix: Numpy array to save
        cache_path: Path where to save the matrix
    """
    np.save(cache_path, matrix)
    print(f"  -> Cached: {cache_path.name}")


def load_matrix(cache_path: Path) -> NDArray[np.floating]:
    """
    Load numpy array from disk.

    Args:
        cache_path: Path to the cached matrix file

    Returns:
        np.ndarray: Loaded matrix
    """
    matrix = np.load(cache_path)
    print(f"  -> Loaded from cache: {cache_path.name}")
    return matrix


def cache_exists(cache_path: Path) -> bool:
    """Check if a cache file exists."""
    return cache_path.exists()


def save_metadata(
    dataset_name: str,
    user_ids: list[Any],
    item_ids: list[Any],
    activity_map: dict[Any, str] | None = None,
) -> None:
    """
    Save metadata (user_ids, item_ids, activity_map) to JSON file.

    Args:
        dataset_name: Name of the dataset
        user_ids: List of user IDs
        item_ids: List of item IDs
        activity_map: Optional dict mapping user_id -> activity group
    """
    cache_dir = get_cache_dir()
    safe_name = _sanitize_dataset_name(dataset_name)
    metadata_path = cache_dir / f"{safe_name}_metadata.json"

    metadata = {
        "user_ids": user_ids,
        "item_ids": item_ids,
        "activity_map": activity_map,
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f)
    print(f"  -> Cached: {metadata_path.name}")


def load_metadata(
    dataset_name: str,
) -> tuple[list[Any], list[Any], dict[Any, str] | None] | None:
    """
    Load metadata (user_ids, item_ids, activity_map) from JSON file.

    Args:
        dataset_name: Name of the dataset

    Returns:
        tuple: (user_ids, item_ids, activity_map) or None if not found
    """
    cache_dir = get_cache_dir()
    safe_name = _sanitize_dataset_name(dataset_name)
    metadata_path = cache_dir / f"{safe_name}_metadata.json"

    if not metadata_path.exists():
        return None

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    print(f"  -> Loaded from cache: {metadata_path.name}")
    return metadata["user_ids"], metadata["item_ids"], metadata.get("activity_map")


def get_all_matrix_cache_paths(dataset_name: str) -> dict[str, Path]:
    """
    Get paths for all matrix cache files for a dataset.

    Args:
        dataset_name: Name of the dataset

    Returns:
        dict: Dictionary with keys 'train', 'val', 'test', 'prediction' and Path values
    """
    return {
        "train": get_cache_path(dataset_name, "train"),
        "val": get_cache_path(dataset_name, "val"),
        "test": get_cache_path(dataset_name, "test"),
        "prediction": get_cache_path(dataset_name, "prediction"),
    }


def all_matrices_cached(dataset_name: str, include_prediction: bool = False) -> bool:
    """
    Check if all required matrices are cached for a dataset.

    Args:
        dataset_name: Name of the dataset
        include_prediction: Whether to also check for prediction matrix

    Returns:
        bool: True if all required caches exist
    """
    paths = get_all_matrix_cache_paths(dataset_name)

    # Check train/val/test matrices
    required = ["train", "val", "test"]
    if include_prediction:
        required.append("prediction")

    # Also check metadata
    cache_dir = get_cache_dir()
    safe_name = _sanitize_dataset_name(dataset_name)
    metadata_path = cache_dir / f"{safe_name}_metadata.json"

    if not metadata_path.exists():
        return False

    return all(cache_exists(paths[key]) for key in required)
