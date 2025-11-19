# Recommendation System Technical Summary

## Evaluation Metrics

### 1. **Precision@K**
- **What it means**: Out of the top K recommendations, how many are actually relevant?
- **Calculation**: `(Number of relevant items in top K) / K`
- **Example**: If top 10 recommendations contain 7 relevant items, Precision@10 = 0.7
- **Use case**: Measures recommendation accuracy at the top of the list

### 2. **Recall@K**
- **What it means**: Out of all relevant items, how many did we find in top K?
- **Calculation**: `(Number of relevant items in top K) / (Total relevant items)`
- **Example**: If there are 20 relevant items total and we found 8 in top 10, Recall@10 = 0.4
- **Use case**: Measures how well we capture all relevant items

### 3. **F1@K**
- **What it means**: Harmonic mean of Precision and Recall (balances both)
- **Calculation**: `2 * (Precision@K * Recall@K) / (Precision@K + Recall@K)`
- **Use case**: Single metric combining precision and recall

### 4. **NDCG@K (Normalized Discounted Cumulative Gain)**
- **What it means**: Measures ranking quality - relevant items should be ranked higher
- **Calculation**: 
  - DCG = Sum of (relevance_score / log2(position + 1)) for top K
  - IDCG = DCG of perfect ranking
  - NDCG = DCG / IDCG
- **Example**: If most relevant items are at top positions, NDCG is high (closer to 1.0)
- **Use case**: Best metric for ranking quality (accounts for position)

### 5. **MAP (Mean Average Precision)**
- **What it means**: Average precision across all relevant items
- **Calculation**: For each relevant item, calculate precision at that position, then average
- **Use case**: Measures overall ranking quality across all relevant items

### 6. **Hit Rate@K**
- **What it means**: Did we recommend at least one relevant item in top K?
- **Calculation**: 1.0 if yes, 0.0 if no (binary metric)
- **Use case**: Simple success metric - did we get anything right?

### 7. **Coverage**
- **What it means**: What percentage of all items have been recommended at least once?
- **Calculation**: `(Unique items recommended) / (Total items in catalog)`
- **Use case**: Measures diversity of recommendations across catalog

**Default K values used**: [10, 20, 50]

---

## Two-Tower Model

### Architecture
- **Two separate neural networks (towers)**:
  - **User Tower**: Encodes user features → embedding vector
  - **Startup Tower**: Encodes startup features → embedding vector
- **Similarity**: Dot product of normalized embeddings → sigmoid → score (0-1)

### Layers
```
User Tower:
  Input (user_feature_dim) 
  → Linear → ReLU → Dropout(0.3)
  → Linear → ReLU → Dropout(0.2)  [if multiple hidden layers]
  → Linear (embedding_dim)
  → L2 Normalize

Startup Tower:
  Input (startup_feature_dim)
  → Linear → ReLU → Dropout(0.3)
  → Linear → ReLU → Dropout(0.2)  [if multiple hidden layers]
  → Linear (embedding_dim)
  → L2 Normalize

Similarity:
  user_emb · startup_emb → sigmoid → score
```

### Hyperparameters
- **Embedding dimension**: 128 (default)
- **Hidden dimensions**: [512, 256] (default) - two hidden layers
- **Dropout rate (first layer)**: 0.3
- **Dropout rate (middle layers)**: 0.2
- **Learning rate**: 0.001
- **Weight decay**: 0.01 (L2 regularization)
- **Batch size**: 256
- **Epochs**: 50
- **Early stopping patience**: 5 epochs
- **Gradient clipping**: 1.0
- **Learning rate scheduler**: Cosine annealing (reduces LR over time)
- **Train/Val/Test split**: 70% / 15% / 15%

### Loss Function
- **Weighted Binary Cross-Entropy (BCE)**
- **Formula**: `BCE(pred, target) * weight` averaged
- **Weights include**:
  - Interaction type weights (view=0.5, click=1.0, like=2.0, apply=3.0, etc.)
  - Position bias correction: `1 / log2(rank + 1)` (higher ranks get lower weight)
- **Purpose**: Prioritizes important interactions and corrects for exposure bias

### Optimizer
- **AdamW** (Adam with weight decay)
- **Learning rate**: 0.001
- **Weight decay**: 0.01

---

## Ranker Model

### Architecture
- **Simple 2-layer MLP (Multi-Layer Perceptron)**
- **Input**: 5 features
  1. `model_score`: Base recommendation score
  2. `recency_score`: How new the startup is
  3. `popularity_score`: Views/interactions count
  4. `diversity_penalty`: Similarity to already ranked items
  5. `original_score`: Teacher signal from base model

### Layers
```
Input (5 features)
  → Linear(5 → 32) → ReLU → Dropout(0.2)
  → Linear(32 → 16) → ReLU
  → Linear(16 → 1)
  → Output: ranking score
```

### Hyperparameters
- **Hidden layer 1**: 32 neurons
- **Hidden layer 2**: 16 neurons
- **Dropout**: 0.2
- **Learning rate**: 0.001
- **Batch size**: 128
- **Epochs**: 20
- **Margin (for loss)**: 1.0

### Loss Function
- **Pairwise Ranking Loss (Margin Loss)**
- **Formula**: `max(0, margin - (pos_score - neg_score))`
- **Weighted by**: Exposure weight (position bias correction)
- **Purpose**: Ensures positive items score higher than negative items by at least the margin

### Optimizer
- **Adam**
- **Learning rate**: 0.001

### Training Data
- **Positive examples**: User liked/applied/favorited from recommendations
- **Negative examples**: User disliked OR high-score recommendation with no interaction (hard negatives)

---

## ALS (Alternating Least Squares) / SVD Model

### Architecture
- **Matrix Factorization using TruncatedSVD** (scikit-learn)
- **Decomposes**: User-Item interaction matrix → User factors × Item factors
- **Method**: Singular Value Decomposition (SVD) on sparse interaction matrix

### How it works
1. Build sparse matrix: Users (rows) × Startups (columns) with interaction weights
2. Apply TruncatedSVD to get low-rank approximation
3. Extract user embeddings and item embeddings
4. Recommendations: Dot product of user embedding with all item embeddings

### Hyperparameters
- **Factors (components)**: 128 (latent dimensions)
- **Iterations**: 10 (power iterations for SVD)
- **Random state**: 42 (for reproducibility)
- **No train/val split by default** (uses all data)

### Loss Function
- **Reconstruction error** (implicit in SVD)
- **Minimizes**: Frobenius norm of (original matrix - reconstructed matrix)
- **SVD automatically finds optimal low-rank approximation**

### Output
- **User factors**: (n_users, 128) - user embeddings
- **Item factors**: (128, n_items) - item embeddings
- **Explained variance**: How much variance is captured by 128 factors

---

## Training Data Preparation

### Interaction Weights
- **View**: 0.5
- **Click**: 1.0
- **Like**: 2.0
- **Dislike**: -1.0 (negative signal)
- **Favorite**: 2.5
- **Apply**: 3.0
- **Interest**: 3.5

### Position Bias Correction
- **Exposure weight**: `1 / log2(rank + 1)`
- **Purpose**: Items shown at top get more clicks, so we down-weight them to avoid bias
- **Example**: Rank 1 → weight 1.0, Rank 10 → weight 0.33, Rank 100 → weight 0.15

### Label Encoding
- **Labels**: 0.0 to 1.0 based on interaction type
- **Higher interaction types** (apply, interest) → higher labels
- **No interaction** → 0.0

---

## Model Comparison

| Model | Type | Input | Output | Use Case |
|-------|------|-------|--------|----------|
| **Two-Tower** | Deep Learning | User features + Startup features | Similarity score (0-1) | Primary recommendation engine |
| **Ranker** | Neural Network | 5 ranking signals | Reranking score | Post-processing to improve ranking |
| **ALS/SVD** | Matrix Factorization | User-Item interactions | Embeddings | Collaborative filtering (users with similar tastes) |

---

## Key Technical Details

### Feature Engineering
- **User features**: Role, preferences, interaction history, embeddings
- **Startup features**: Category, type, description embeddings, metadata
- **Feature encoder**: Handles categorical encoding, normalization, missing values

### Training Strategy
- **Two-Tower**: Trained on positive and negative pairs (hard negatives)
- **Ranker**: Trained on pairwise comparisons (positive vs negative)
- **ALS**: Trained on implicit feedback matrix (all interactions)

### Evaluation Strategy
- **Metrics computed**: Precision@K, Recall@K, F1@K, NDCG@K, MAP, Hit Rate@K
- **K values**: [10, 20, 50]
- **Validation**: Every epoch (or every N epochs)
- **Early stopping**: Based on NDCG@10 improvement

### Model Selection
- **Best model**: Selected based on highest NDCG@10 on validation set
- **Checkpointing**: Saves best model and latest model
- **Config saved**: All hyperparameters saved as JSON

---

## Summary

**Metrics**: Precision@K, Recall@K, F1@K, NDCG@K, MAP, Hit Rate@K, Coverage
- **NDCG@K is the primary metric** for ranking quality

**Two-Tower**: Deep neural network with two towers, embedding dimension 128, hidden layers [512, 256], weighted BCE loss

**Ranker**: Simple 2-layer MLP (32→16→1), pairwise ranking loss with margin 1.0

**ALS**: Matrix factorization with 128 factors using TruncatedSVD

**All models use**: Adam/AdamW optimizer, learning rate 0.001, dropout for regularization, early stopping

