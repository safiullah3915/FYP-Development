"""
SVD-based Collaborative Filtering Training Script
Replaces the implicit ALS dependency with TruncatedSVD (scikit-learn)
"""
import argparse
import os
import json
import time
from datetime import datetime

import numpy as np
from scipy.sparse import load_npz, csr_matrix
from sklearn.decomposition import TruncatedSVD


def load_data(data_path, user_mapping_path, item_mapping_path):
    """Load sparse interaction matrix and ID mappings"""
    print(f"\nLoading data from {data_path}...")

    interactions = load_npz(data_path).tocsr()
    print(f"  Matrix shape: {interactions.shape}")
    print(f"  Non-zero entries: {interactions.nnz}")
    density = interactions.nnz / (interactions.shape[0] * interactions.shape[1])
    print(f"  Density: {density * 100:.4f}%")

    with open(user_mapping_path, 'r') as f:
        user_mapping = json.load(f)

    with open(item_mapping_path, 'r') as f:
        item_mapping = json.load(f)

    print(f"  Users: {len(user_mapping)}")
    print(f"  Items: {len(item_mapping)}")

    return interactions, user_mapping, item_mapping


def create_train_test_split(interactions, test_ratio=0.2, seed=42):
    """Create a basic train/test split for diagnostics"""
    if test_ratio <= 0:
        return interactions, None

    print(f"\nCreating train/test split (test_ratio={test_ratio})...")

    rng = np.random.default_rng(seed)
    interactions = interactions.tocoo()

    n_interactions = interactions.nnz
    indices = np.arange(n_interactions)
    rng.shuffle(indices)

    n_test = int(n_interactions * test_ratio)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    train_matrix = csr_matrix(
        (interactions.data[train_indices],
         (interactions.row[train_indices], interactions.col[train_indices])),
        shape=interactions.shape
    )
    test_matrix = csr_matrix(
        (interactions.data[test_indices],
         (interactions.row[test_indices], interactions.col[test_indices])),
        shape=interactions.shape
    )

    print(f"  Train interactions: {train_matrix.nnz}")
    print(f"  Test interactions: {test_matrix.nnz}")

    return train_matrix, test_matrix


def train_svd_model(interactions, factors=128, n_iter=10, random_state=42):
    """Train TruncatedSVD on the implicit interaction matrix"""
    print(f"\n=== Training SVD Model ===")
    print("Hyperparameters:")
    print(f"  components (factors): {factors}")
    print(f"  n_iter: {n_iter}")
    print(f"  random_state: {random_state}")

    model = TruncatedSVD(
        n_components=factors,
        n_iter=n_iter,
        random_state=random_state,
    )

    start_time = time.time()
    user_latent = model.fit_transform(interactions)  # U * Sigma
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")

    singular_values = model.singular_values_
    sqrt_sigma = np.sqrt(np.maximum(singular_values, 1e-8))

    user_factors = user_latent / sqrt_sigma[np.newaxis, :]
    item_factors = (model.components_.T * sqrt_sigma[np.newaxis, :])

    return {
        'user_factors': user_factors.astype(np.float32),
        'item_factors': item_factors.astype(np.float32),
        'singular_values': singular_values,
        'explained_variance': float(model.explained_variance_ratio_.sum()),
        'model': model,
    }


def evaluate_model(model_dict, train_matrix, test_matrix):
    """Provide lightweight diagnostics since implicit metrics are removed"""
    if test_matrix is None:
        return {}

    print(f"\n=== Evaluating Model (diagnostic) ===")
    print(f"  Explained variance ratio: {model_dict['explained_variance']:.4f}")

    # Reconstruction error on a sampled subset of rows
    rng = np.random.default_rng(42)
    n_rows = min(1000, train_matrix.shape[0])
    sample_indices = rng.choice(train_matrix.shape[0], size=n_rows, replace=False)
    sampled = train_matrix[sample_indices].toarray()

    approx = model_dict['model'].inverse_transform(
        model_dict['model'].transform(train_matrix[sample_indices])
    )
    mse = float(np.mean((sampled - approx) ** 2))
    print(f"  Sampled reconstruction MSE: {mse:.6f}")
    return {
        'explained_variance': model_dict['explained_variance'],
        'sampled_reconstruction_mse': mse,
    }


def save_artifacts(model_dict, user_mapping, item_mapping, output_dir, model_name="als_v1"):
    """Persist embeddings, mappings, and config metadata"""
    print(f"\n=== Saving Model Artifacts ===")
    os.makedirs(output_dir, exist_ok=True)

    prefix = os.path.join(output_dir, model_name)

    user_factors_path = f"{prefix}_user_factors.npy"
    np.save(user_factors_path, model_dict['user_factors'])
    print(f"Saved user factors: {user_factors_path}")

    item_factors_path = f"{prefix}_item_factors.npy"
    np.save(item_factors_path, model_dict['item_factors'])
    print(f"Saved item factors: {item_factors_path}")

    user_mapping_path = f"{prefix}_user_mapping.json"
    with open(user_mapping_path, 'w') as f:
        json.dump(user_mapping, f)
    print(f"Saved user mapping: {user_mapping_path}")

    item_mapping_path = f"{prefix}_item_mapping.json"
    with open(item_mapping_path, 'w') as f:
        json.dump(item_mapping, f)
    print(f"Saved item mapping: {item_mapping_path}")

    config_path = f"{prefix}_config.json"
    config_payload = {
        "algorithm": "svd",
        "library": "scikit-learn",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "n_users": len(user_mapping),
        "n_items": len(item_mapping),
        "n_components": int(model_dict['user_factors'].shape[1]),
        "explained_variance": model_dict['explained_variance'],
        "singular_values": model_dict['singular_values'].tolist(),
    }
    with open(config_path, 'w') as f:
        json.dump(config_payload, f, indent=2)
    print(f"Saved config: {config_path}")


def main():
    parser = argparse.ArgumentParser(description='Train SVD collaborative filtering model')
    parser.add_argument('--data', type=str, default='data/als_interactions.npz',
                        help='Path to interaction matrix file')
    parser.add_argument('--user-mapping', type=str, default='data/als_user_mapping.json',
                        help='Path to user mapping file')
    parser.add_argument('--item-mapping', type=str, default='data/als_item_mapping.json',
                        help='Path to item mapping file')
    parser.add_argument('--output-dir', type=str, default='models',
                        help='Output directory for trained model')
    parser.add_argument('--model-name', type=str, default='als_v1',
                        help='Model name prefix for saving artifacts')
    parser.add_argument('--factors', type=int, default=128,
                        help='Number of latent factors (SVD components)')
    parser.add_argument('--iterations', type=int, default=10,
                        help='Number of power iterations for TruncatedSVD')
    parser.add_argument('--random-state', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--test-split', type=float, default=0.0,
                        help='Optional test split ratio for diagnostics')
    parser.add_argument('--no-eval', action='store_true',
                        help='Skip diagnostics')

    args = parser.parse_args()

    print("=" * 60)
    print("SVD COLLABORATIVE FILTERING TRAINING")
    print("=" * 60)

    interactions, user_mapping, item_mapping = load_data(
        args.data, args.user_mapping, args.item_mapping
    )

    if not args.no_eval and args.test_split > 0:
        train_matrix, test_matrix = create_train_test_split(
            interactions, test_ratio=args.test_split
        )
    else:
        train_matrix, test_matrix = interactions, None

    model_dict = train_svd_model(
        train_matrix,
        factors=args.factors,
        n_iter=args.iterations,
        random_state=args.random_state,
    )

    if not args.no_eval:
        evaluate_model(model_dict, train_matrix, test_matrix)

    save_artifacts(model_dict, user_mapping, item_mapping, args.output_dir, args.model_name)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nArtifacts saved with prefix: {args.model_name}")
    print("Deploy by copying *.npy, *_mapping.json, and *_config.json to the models directory.")
    print()


if __name__ == '__main__':
    main()
