"""
Script to verify reverse models are loaded and working
"""
import sys
from pathlib import Path
import json

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from app import (
    two_tower_reverse_model,
    als_reverse_model,
    ranker_reverse_model,
    MODELS_DIR,
    TWO_TOWER_REVERSE_MODEL_PATH,
    RANKER_REVERSE_MODEL_PATH,
    ALS_REVERSE_MODEL_NAME
)

def check_model_files():
    """Check if model files exist"""
    print("=" * 60)
    print("CHECKING MODEL FILES")
    print("=" * 60)
    
    models_status = {}
    
    # Check Two-Tower Reverse
    two_tower_exists = TWO_TOWER_REVERSE_MODEL_PATH.exists()
    models_status['two_tower_reverse'] = {
        'file_exists': two_tower_exists,
        'path': str(TWO_TOWER_REVERSE_MODEL_PATH),
        'size_mb': TWO_TOWER_REVERSE_MODEL_PATH.stat().st_size / (1024 * 1024) if two_tower_exists else 0
    }
    
    # Check encoder
    encoder_path = MODELS_DIR / "two_tower_reverse_v1_encoder.pkl"
    models_status['two_tower_reverse']['encoder_exists'] = encoder_path.exists()
    models_status['two_tower_reverse']['encoder_path'] = str(encoder_path)
    
    # Check config
    config_path = MODELS_DIR / "two_tower_reverse_v1_config.json"
    models_status['two_tower_reverse']['config_exists'] = config_path.exists()
    models_status['two_tower_reverse']['config_path'] = str(config_path)
    
    # Check Ranker Reverse
    ranker_exists = RANKER_REVERSE_MODEL_PATH.exists()
    models_status['ranker_reverse'] = {
        'file_exists': ranker_exists,
        'path': str(RANKER_REVERSE_MODEL_PATH),
        'size_mb': RANKER_REVERSE_MODEL_PATH.stat().st_size / (1024 * 1024) if ranker_exists else 0
    }
    
    # Check ALS Reverse
    als_config_path = MODELS_DIR / f"{ALS_REVERSE_MODEL_NAME}_config.json"
    als_user_factors = MODELS_DIR / f"{ALS_REVERSE_MODEL_NAME}_user_factors.npy"
    als_item_factors = MODELS_DIR / f"{ALS_REVERSE_MODEL_NAME}_item_factors.npy"
    
    models_status['als_reverse'] = {
        'config_exists': als_config_path.exists(),
        'user_factors_exists': als_user_factors.exists(),
        'item_factors_exists': als_item_factors.exists(),
        'config_path': str(als_config_path),
        'user_factors_path': str(als_user_factors),
        'item_factors_path': str(als_item_factors),
    }
    
    # Print status
    for model_name, status in models_status.items():
        print(f"\n{model_name.upper()}:")
        for key, value in status.items():
            if 'exists' in key:
                symbol = "✓" if value else "✗"
                print(f"  {symbol} {key}: {value}")
            elif 'path' in key:
                print(f"  📁 {key}: {value}")
            elif 'size' in key:
                print(f"  📦 {key}: {value:.2f} MB")
    
    return models_status

def check_model_loading():
    """Check if models are loaded in memory"""
    print("\n" + "=" * 60)
    print("CHECKING MODEL LOADING")
    print("=" * 60)
    
    loading_status = {}
    
    # Check Two-Tower Reverse
    loading_status['two_tower_reverse'] = {
        'loaded': two_tower_reverse_model is not None,
        'type': type(two_tower_reverse_model).__name__ if two_tower_reverse_model else None
    }
    
    # Check ALS Reverse
    loading_status['als_reverse'] = {
        'loaded': als_reverse_model is not None,
        'type': type(als_reverse_model).__name__ if als_reverse_model else None
    }
    
    # Check Ranker Reverse
    loading_status['ranker_reverse'] = {
        'loaded': ranker_reverse_model is not None,
        'type': type(ranker_reverse_model).__name__ if ranker_reverse_model else None
    }
    
    # Print status
    for model_name, status in loading_status.items():
        symbol = "✓" if status['loaded'] else "✗"
        print(f"\n{symbol} {model_name.upper()}:")
        print(f"    Loaded: {status['loaded']}")
        if status['loaded']:
            print(f"    Type: {status['type']}")
    
    return loading_status

def test_inference():
    """Test model inference with a sample startup"""
    print("\n" + "=" * 60)
    print("TESTING INFERENCE")
    print("=" * 60)
    
    # Get a sample startup ID from database
    try:
        from database.connection import SessionLocal
        from database.models import Startup
        
        db = SessionLocal()
        startup = db.query(Startup).filter(Startup.status == 'active').first()
        db.close()
        
        if not startup:
            print("⚠ No active startups found in database")
            return
        
        startup_id = str(startup.id)
        print(f"\nTesting with startup: {startup.title} (ID: {startup_id})")
        
        # Test Two-Tower Reverse
        if two_tower_reverse_model:
            try:
                print("\n🔍 Testing Two-Tower Reverse Model...")
                results = two_tower_reverse_model.recommend(startup_id, limit=5)
                print(f"  ✓ Success! Returned {len(results.get('developers', []))} developers")
                print(f"  Method: {results.get('method', 'unknown')}")
                print(f"  Model Version: {results.get('model_version', 'unknown')}")
            except Exception as e:
                print(f"  ✗ Error: {e}")
        else:
            print("\n✗ Two-Tower Reverse Model not loaded")
        
        # Test ALS Reverse
        if als_reverse_model:
            try:
                print("\n🔍 Testing ALS Reverse Model...")
                results = als_reverse_model.recommend(startup_id, limit=5)
                print(f"  ✓ Success! Returned {len(results.get('developers', []))} developers")
                print(f"  Method: {results.get('method', 'unknown')}")
                print(f"  Model Version: {results.get('model_version', 'unknown')}")
            except Exception as e:
                print(f"  ✗ Error: {e}")
        else:
            print("\n✗ ALS Reverse Model not loaded")
        
        # Test Ranker Reverse
        if ranker_reverse_model:
            try:
                print("\n🔍 Testing Ranker Reverse Model...")
                # Create dummy candidates
                candidates = [
                    {'user_id': 'test1', 'score': 0.8},
                    {'user_id': 'test2', 'score': 0.7},
                    {'user_id': 'test3', 'score': 0.6},
                ]
                reranked = ranker_reverse_model.rank(
                    candidates=candidates,
                    user_id=startup_id,
                    already_ranked=[]
                )
                print(f"  ✓ Success! Ranked {len(reranked)} candidates")
            except Exception as e:
                print(f"  ✗ Error: {e}")
        else:
            print("\n✗ Ranker Reverse Model not loaded")
            
    except Exception as e:
        print(f"⚠ Could not test inference: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Run all checks"""
    print("\n" + "=" * 60)
    print("REVERSE MODEL VERIFICATION")
    print("=" * 60)
    
    # Check files
    file_status = check_model_files()
    
    # Check loading
    loading_status = check_model_loading()
    
    # Test inference
    test_inference()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_loaded = all(status['loaded'] for status in loading_status.values())
    all_files_exist = (
        file_status['two_tower_reverse']['file_exists'] and
        file_status['ranker_reverse']['file_exists'] and
        (file_status['als_reverse']['config_exists'] or 
         file_status['als_reverse']['user_factors_exists'])
    )
    
    if all_loaded and all_files_exist:
        print("✅ All reverse models are loaded and ready!")
    elif all_files_exist:
        print("⚠ Model files exist but some models failed to load")
        print("   Check Flask app logs for loading errors")
    else:
        print("❌ Some model files are missing")
        print("   Make sure you've trained all reverse models")
    
    print("\nTo test via API:")
    print("  1. Start Flask app: python app.py")
    print("  2. Call endpoint: GET /api/recommendations/developers/for-startup/<startup_id>")
    print("  3. Check response for 'method_used' and 'model_version' fields")

if __name__ == "__main__":
    main()

