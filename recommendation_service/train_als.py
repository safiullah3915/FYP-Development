"""
ALS Model Training Script
Trains Alternating Least Squares collaborative filtering model
"""
import argparse
import os
import json
import pickle
import numpy as np
from scipy.sparse import load_npz
from implicit.als import AlternatingLeastSquares
from implicit.evaluation import precision_at_k, mean_average_precision_at_k
import time


def load_data(data_path, user_mapping_path, item_mapping_path):
    """Load sparse interaction matrix and ID mappings"""
    print(f"\nLoading data from {data_path}...")
    
    # Load sparse matrix
    interactions = load_npz(data_path)
    print(f"  Matrix shape: {interactions.shape}")
    print(f"  Non-zero entries: {interactions.nnz}")
    print(f"  Density: {interactions.nnz / (interactions.shape[0] * interactions.shape[1]) * 100:.4f}%")
    
    # Load mappings
    with open(user_mapping_path, 'r') as f:
        user_mapping = json.load(f)
    
    with open(item_mapping_path, 'r') as f:
        item_mapping = json.load(f)
    
    # Create reverse mappings
    user_reverse = {int(v): k for k, v in user_mapping.items()}
    item_reverse = {int(v): k for k, v in item_mapping.items()}
    
    print(f"  Users: {len(user_mapping)}")
    print(f"  Items: {len(item_mapping)}")
    
    return interactions, user_mapping, item_mapping, user_reverse, item_reverse


def create_train_test_split(interactions, test_ratio=0.2, seed=42):
    """Create train/test split for evaluation"""
    print(f"\nCreating train/test split (test_ratio={test_ratio})...")
    
    np.random.seed(seed)
    interactions = interactions.tocoo()
    
    # Get all non-zero entries
    n_interactions = interactions.nnz
    indices = np.arange(n_interactions)
    np.random.shuffle(indices)
    
    # Split indices
    n_test = int(n_interactions * test_ratio)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    # Create train matrix
    train_data = interactions.data[train_indices]
    train_row = interactions.row[train_indices]
    train_col = interactions.col[train_indices]
    
    from scipy.sparse import csr_matrix
    train_matrix = csr_matrix(
        (train_data, (train_row, train_col)),
        shape=interactions.shape
    )
    
    # Create test matrix
    test_data = interactions.data[test_indices]
    test_row = interactions.row[test_indices]
    test_col = interactions.col[test_indices]
    
    test_matrix = csr_matrix(
        (test_data, (test_row, test_col)),
        shape=interactions.shape
    )
    
    print(f"  Train interactions: {train_matrix.nnz}")
    print(f"  Test interactions: {test_matrix.nnz}")
    
    return train_matrix, test_matrix


def train_als_model(interactions, factors=128, regularization=0.01, iterations=20, alpha=40):
    """Train ALS model"""
    print(f"\n=== Training ALS Model ===")
    print(f"Hyperparameters:")
    print(f"  factors: {factors}")
    print(f"  regularization: {regularization}")
    print(f"  iterations: {iterations}")
    print(f"  alpha: {alpha}")
    
    # Initialize model
    model = AlternatingLeastSquares(
        factors=factors,
        regularization=regularization,
        iterations=iterations,
        alpha=alpha,
        random_state=42
    )
    
    # Train model
    print(f"\nTraining...")
    start_time = time.time()
    
    # ALS expects item-user matrix (items x users), so we transpose
    model.fit(interactions.T)
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    return model


def evaluate_model(model, train_matrix, test_matrix, k=10):
    """Evaluate model on test set"""
    print(f"\n=== Evaluating Model ===")
    
    try:
        # Calculate Precision@K
        precision = precision_at_k(model, train_matrix, test_matrix, K=k)
        print(f"Precision@{k}: {precision:.4f}")
        
        # Calculate MAP@K
        map_score = mean_average_precision_at_k(model, train_matrix, test_matrix, K=k)
        print(f"MAP@{k}: {map_score:.4f}")
        
        return {
            f'precision@{k}': precision,
            f'map@{k}': map_score
        }
    except Exception as e:
        print(f"Warning: Evaluation failed: {e}")
        return {}


def save_model(model, user_mapping, item_mapping, output_dir, model_name="als_v1"):
    """Save trained model and embeddings"""
    print(f"\n=== Saving Model ===")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save full model
    model_path = os.path.join(output_dir, f"{model_name}.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Saved model: {model_path}")
    
    # Save user factors (embeddings)
    user_factors_path = os.path.join(output_dir, f"{model_name}_user_factors.npy")
    np.save(user_factors_path, model.user_factors)
    print(f"Saved user factors: {user_factors_path}")
    
    # Save item factors (embeddings)
    item_factors_path = os.path.join(output_dir, f"{model_name}_item_factors.npy")
    np.save(item_factors_path, model.item_factors)
    print(f"Saved item factors: {item_factors_path}")
    
    # Save mappings alongside model
    user_mapping_path = os.path.join(output_dir, f"{model_name}_user_mapping.json")
    with open(user_mapping_path, 'w') as f:
        json.dump(user_mapping, f)
    print(f"Saved user mapping: {user_mapping_path}")
    
    item_mapping_path = os.path.join(output_dir, f"{model_name}_item_mapping.json")
    with open(item_mapping_path, 'w') as f:
        json.dump(item_mapping, f)
    print(f"Saved item mapping: {item_mapping_path}")
    
    print(f"\nModel artifacts saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Train ALS collaborative filtering model')
    parser.add_argument('--data', type=str, default='data/als_interactions.npz',
                        help='Path to interaction matrix file')
    parser.add_argument('--user-mapping', type=str, default='data/als_user_mapping.json',
                        help='Path to user mapping file')
    parser.add_argument('--item-mapping', type=str, default='data/als_item_mapping.json',
                        help='Path to item mapping file')
    parser.add_argument('--output-dir', type=str, default='models',
                        help='Output directory for trained model')
    parser.add_argument('--model-name', type=str, default='als_v1',
                        help='Model name for saving')
    parser.add_argument('--factors', type=int, default=128,
                        help='Number of latent factors')
    parser.add_argument('--regularization', type=float, default=0.01,
                        help='Regularization parameter')
    parser.add_argument('--iterations', type=int, default=20,
                        help='Number of training iterations')
    parser.add_argument('--alpha', type=float, default=40,
                        help='Confidence weight for implicit feedback')
    parser.add_argument('--test-split', type=float, default=0.2,
                        help='Test set ratio for evaluation')
    parser.add_argument('--no-eval', action='store_true',
                        help='Skip evaluation')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("ALS COLLABORATIVE FILTERING TRAINING")
    print("=" * 60)
    
    # Load data
    interactions, user_mapping, item_mapping, user_reverse, item_reverse = load_data(
        args.data, args.user_mapping, args.item_mapping
    )
    
    # Split data if evaluation is enabled
    if not args.no_eval:
        train_matrix, test_matrix = create_train_test_split(
            interactions, test_ratio=args.test_split
        )
    else:
        train_matrix = interactions
        test_matrix = None
    
    # Train model
    model = train_als_model(
        train_matrix,
        factors=args.factors,
        regularization=args.regularization,
        iterations=args.iterations,
        alpha=args.alpha
    )
    
    # Evaluate if test set exists
    if test_matrix is not None:
        metrics = evaluate_model(model, train_matrix, test_matrix)
    
    # Save model
    save_model(model, user_mapping, item_mapping, args.output_dir, args.model_name)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nTo use this model:")
    print(f"  1. Make sure Flask app has access to {args.output_dir}/")
    print(f"  2. Model will be loaded automatically on startup")
    print(f"  3. ALS will be used for warm users (5-19 interactions)")
    print(f"  4. Ensemble will use ALS for hot users (20+ interactions)")
    print()


if __name__ == '__main__':
    main()


