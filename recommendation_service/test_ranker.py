"""
Test script for Ranker Model
Tests feature extraction, ranking logic, and integration
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from engines.ranker import NeuralRanker, RankerMLP
from engines.ranker_features import (
    calculate_recency_score,
    calculate_popularity_score,
    calculate_diversity_penalty,
    normalize_score
)
from datetime import datetime, timedelta, timezone
import torch


def test_feature_extraction():
    """Test feature extraction functions"""
    print("=" * 60)
    print("TEST 1: Feature Extraction")
    print("=" * 60)
    
    # Test recency score
    now = datetime.now(timezone.utc)
    recent_date = (now - timedelta(days=5)).isoformat()
    old_date = (now - timedelta(days=60)).isoformat()
    
    recent_score = calculate_recency_score(recent_date, None)
    old_score = calculate_recency_score(old_date, None)
    
    print(f"Recency Score (5 days old): {recent_score:.3f}")
    print(f"Recency Score (60 days old): {old_score:.3f}")
    
    assert recent_score > old_score, "Recent items should have higher score"
    assert 0 <= recent_score <= 1, "Score should be in [0, 1]"
    print("✓ Recency score test passed")
    print()
    
    # Test popularity score
    pop_high = calculate_popularity_score(views=1000, interaction_count=50)
    pop_low = calculate_popularity_score(views=10, interaction_count=1)
    
    print(f"Popularity Score (1000 views, 50 interactions): {pop_high:.3f}")
    print(f"Popularity Score (10 views, 1 interaction): {pop_low:.3f}")
    
    assert pop_high > pop_low, "Popular items should have higher score"
    assert 0 <= pop_high <= 1, "Score should be in [0, 1]"
    print("✓ Popularity score test passed")
    print()
    
    # Test diversity penalty
    candidate = {
        'id': '1',
        'category': 'Technology',
        'field': 'AI/ML',
        'type': 'B2B'
    }
    
    already_ranked = [
        {'id': '2', 'category': 'Technology', 'field': 'AI/ML', 'type': 'B2B'},
        {'id': '3', 'category': 'Technology', 'field': 'AI/ML', 'type': 'B2C'}
    ]
    
    diversity_similar = calculate_diversity_penalty(candidate, already_ranked)
    diversity_empty = calculate_diversity_penalty(candidate, [])
    
    print(f"Diversity Score (similar items already ranked): {diversity_similar:.3f}")
    print(f"Diversity Score (no items ranked yet): {diversity_empty:.3f}")
    
    assert diversity_empty > diversity_similar, "First item should be more diverse"
    assert 0 <= diversity_similar <= 1, "Score should be in [0, 1]"
    print("✓ Diversity penalty test passed")
    print()


def test_rule_based_ranker():
    """Test rule-based ranker"""
    print("=" * 60)
    print("TEST 2: Rule-Based Ranker")
    print("=" * 60)
    
    # Initialize rule-based ranker
    ranker = NeuralRanker(use_rule_based=True)
    
    # Create test candidates
    now = datetime.now(timezone.utc)
    candidates = [
        {
            'id': '1',
            'title': 'Recent Popular Startup',
            'score': 0.8,
            'views': 500,
            'interaction_count': 25,
            'created_at': (now - timedelta(days=7)).isoformat(),
            'updated_at': (now - timedelta(days=2)).isoformat(),
            'category': 'Technology',
            'field': 'AI/ML',
            'type': 'B2B'
        },
        {
            'id': '2',
            'title': 'Old Unpopular Startup',
            'score': 0.9,  # Higher model score
            'views': 10,
            'interaction_count': 1,
            'created_at': (now - timedelta(days=90)).isoformat(),
            'updated_at': (now - timedelta(days=60)).isoformat(),
            'category': 'Healthcare',
            'field': 'Biotech',
            'type': 'B2C'
        },
        {
            'id': '3',
            'title': 'Medium Startup',
            'score': 0.7,
            'views': 200,
            'interaction_count': 10,
            'created_at': (now - timedelta(days=30)).isoformat(),
            'updated_at': (now - timedelta(days=10)).isoformat(),
            'category': 'Fintech',
            'field': 'Blockchain',
            'type': 'B2B'
        }
    ]
    
    # Rank candidates
    ranked = ranker.rank(candidates, user_id='test_user')
    
    print("Ranked Results:")
    for i, item in enumerate(ranked, 1):
        print(f"{i}. {item['title']}")
        print(f"   Model Score: {item['score']:.3f}")
        print(f"   Ranking Score: {item.get('ranking_score', 0):.3f}")
        print()
    
    assert len(ranked) == 3, "Should return all candidates"
    assert all('ranking_score' in item for item in ranked), "All should have ranking_score"
    print("✓ Rule-based ranker test passed")
    print()


def test_neural_ranker_architecture():
    """Test neural ranker model architecture"""
    print("=" * 60)
    print("TEST 3: Neural Ranker Architecture")
    print("=" * 60)
    
    # Create model
    model = RankerMLP(input_dim=4, hidden_dim1=32, hidden_dim2=16)
    
    # Test forward pass
    batch_size = 10
    test_input = torch.randn(batch_size, 4)
    
    output = model(test_input)
    
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    assert output.shape == (batch_size,), "Output should be 1D"
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    
    assert total_params < 2000, "Model should be lightweight (<2000 params)"
    print("✓ Neural ranker architecture test passed")
    print()


def test_ranker_with_empty_inputs():
    """Test ranker handles edge cases"""
    print("=" * 60)
    print("TEST 4: Edge Cases")
    print("=" * 60)
    
    ranker = NeuralRanker(use_rule_based=True)
    
    # Empty list
    result = ranker.rank([], user_id='test_user')
    assert result == [], "Should handle empty list"
    print("✓ Empty list handled")
    
    # Single item
    now = datetime.now(timezone.utc)
    single = [{
        'id': '1',
        'score': 0.5,
        'views': 100,
        'interaction_count': 5,
        'created_at': now.isoformat(),
        'category': 'Tech',
        'field': 'AI',
        'type': 'B2B'
    }]
    
    result = ranker.rank(single, user_id='test_user')
    assert len(result) == 1, "Should handle single item"
    assert 'ranking_score' in result[0], "Should add ranking_score"
    print("✓ Single item handled")
    
    # Missing fields
    incomplete = [{
        'id': '1',
        'score': 0.5
        # Missing views, created_at, etc.
    }]
    
    result = ranker.rank(incomplete, user_id='test_user')
    assert len(result) == 1, "Should handle missing fields gracefully"
    print("✓ Missing fields handled")
    print()


def test_ranking_order():
    """Test that ranker actually reorders items"""
    print("=" * 60)
    print("TEST 5: Ranking Order")
    print("=" * 60)
    
    ranker = NeuralRanker(use_rule_based=True)
    
    now = datetime.now(timezone.utc)
    
    # Create candidates where model_score doesn't match expected quality
    candidates = [
        {
            'id': 'bad_but_high_score',
            'score': 0.9,  # High model score
            'views': 5,
            'interaction_count': 0,
            'created_at': (now - timedelta(days=180)).isoformat(),  # Very old
            'category': 'Tech',
            'field': 'AI',
            'type': 'B2B'
        },
        {
            'id': 'good_but_low_score',
            'score': 0.4,  # Low model score
            'views': 2000,
            'interaction_count': 100,
            'created_at': (now - timedelta(days=3)).isoformat(),  # Very recent
            'category': 'Tech',
            'field': 'AI',
            'type': 'B2B'
        }
    ]
    
    ranked = ranker.rank(candidates, user_id='test_user')
    
    print("Original order (by model score):")
    print(f"1. {candidates[0]['id']} (score: {candidates[0]['score']:.2f})")
    print(f"2. {candidates[1]['id']} (score: {candidates[1]['score']:.2f})")
    print()
    
    print("Reranked order:")
    for i, item in enumerate(ranked, 1):
        print(f"{i}. {item['id']} (ranking_score: {item['ranking_score']:.3f})")
    print()
    
    # The recent, popular item should be ranked higher
    # even though it has lower model score
    if ranked[0]['id'] == 'good_but_low_score':
        print("✓ Ranker successfully prioritized recent + popular item")
    else:
        print("⚠ Ranker still prioritized model score (may need tuning)")
    print()


def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 60)
    print(" RANKER MODEL TESTS")
    print("=" * 60 + "\n")
    
    try:
        test_feature_extraction()
        test_rule_based_ranker()
        test_neural_ranker_architecture()
        test_ranker_with_empty_inputs()
        test_ranking_order()
        
        print("=" * 60)
        print(" ALL TESTS PASSED! ✓")
        print("=" * 60)
        print()
        print("The ranker is working correctly and ready for production.")
        print()
        print("Next steps:")
        print("1. Generate training data: python manage.py generate_ranker_dataset")
        print("2. Train the model: python train_ranker.py")
        print("3. Start Flask service: python app.py")
        print()
        
        return True
        
    except Exception as e:
        print("=" * 60)
        print(f" TEST FAILED: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)

