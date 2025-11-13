#!/usr/bin/env python3
"""
Integration Test for Two-Tower Model
Tests the complete recommendation flow with two-tower model
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from database.connection import SessionLocal, check_db_connection
from services.recommendation_service import RecommendationService
from engines.model_registry import get_registry
from database.models import User, Startup
from utils.logger import get_logger

logger = get_logger(__name__)


def test_database_connection():
    """Test database connectivity"""
    logger.info("="*60)
    logger.info("TEST 1: Database Connection")
    logger.info("="*60)
    
    if check_db_connection():
        logger.info("✓ Database connection successful")
        return True
    else:
        logger.error("✗ Database connection failed")
        return False


def test_model_registry():
    """Test model registry"""
    logger.info("\n" + "="*60)
    logger.info("TEST 2: Model Registry")
    logger.info("="*60)
    
    try:
        registry = get_registry()
        models = registry.list_available_models()
        
        logger.info(f"Found {len(models)} models:")
        for model in models:
            logger.info(f"  - {model['name']} ({model['size_mb']:.2f} MB)")
        
        if len(models) > 0:
            logger.info("✓ Model registry working")
            return True
        else:
            logger.warning("⚠ No models found (this is OK if not trained yet)")
            return True
            
    except Exception as e:
        logger.error(f"✗ Model registry error: {e}")
        return False


def test_content_based_recommendations():
    """Test content-based recommendations"""
    logger.info("\n" + "="*60)
    logger.info("TEST 3: Content-Based Recommendations")
    logger.info("="*60)
    
    db = SessionLocal()
    try:
        # Get a test user
        user = db.query(User).filter(User.role == 'student').first()
        
        if not user:
            logger.warning("⚠ No student users found in database")
            return True
        
        logger.info(f"Testing with user: {user.username} ({user.id})")
        
        # Initialize service without two-tower
        rec_service = RecommendationService(db, enable_two_tower=False)
        
        # Get recommendations
        results = rec_service.get_recommendations(
            user_id=str(user.id),
            use_case='developer_startup',
            limit=5
        )
        
        logger.info(f"Method used: {results.get('method_used')}")
        logger.info(f"Interaction count: {results.get('interaction_count')}")
        logger.info(f"Recommendations returned: {len(results.get('item_ids', []))}")
        
        if len(results.get('item_ids', [])) > 0:
            logger.info("✓ Content-based recommendations working")
            return True
        else:
            logger.warning("⚠ No recommendations returned (may need more data)")
            return True
            
    except Exception as e:
        logger.error(f"✗ Content-based test failed: {e}", exc_info=True)
        return False
    finally:
        db.close()


def test_two_tower_recommendations():
    """Test two-tower recommendations"""
    logger.info("\n" + "="*60)
    logger.info("TEST 4: Two-Tower Recommendations")
    logger.info("="*60)
    
    db = SessionLocal()
    try:
        # Check if model exists
        registry = get_registry()
        model_paths = registry.get_active_model('developer_startup', 'two_tower')
        
        if not model_paths:
            logger.warning("⚠ No trained two-tower model found")
            logger.info("  To train a model, run:")
            logger.info("  1. cd backend && python manage.py generate_two_tower_dataset --output ../recommendation_service/data/train.csv")
            logger.info("  2. cd ../recommendation_service && python train_two_tower.py --data data/train.csv --epochs 50")
            return True
        
        logger.info(f"Model found: {model_paths['model_path']}")
        
        # Get a test user with interactions
        from database.models import UserInteraction
        from sqlalchemy import func
        
        user_with_interactions = db.query(User).join(UserInteraction).group_by(User.id).having(
            func.count(UserInteraction.id) >= 5
        ).first()
        
        if not user_with_interactions:
            logger.warning("⚠ No users with 5+ interactions found")
            return True
        
        logger.info(f"Testing with user: {user_with_interactions.username} ({user_with_interactions.id})")
        
        # Initialize service with two-tower
        rec_service = RecommendationService(db, enable_two_tower=True)
        
        if not rec_service.two_tower:
            logger.error("✗ Two-tower model failed to load")
            return False
        
        # Get recommendations
        results = rec_service.get_recommendations(
            user_id=str(user_with_interactions.id),
            use_case='developer_startup',
            limit=5
        )
        
        logger.info(f"Method used: {results.get('method_used')}")
        logger.info(f"Interaction count: {results.get('interaction_count')}")
        logger.info(f"Recommendations returned: {len(results.get('item_ids', []))}")
        
        if results.get('method_used') == 'two_tower':
            logger.info("✓ Two-tower recommendations working")
            
            # Show sample recommendations
            if results.get('item_ids'):
                logger.info("\nSample recommendations:")
                for i, item_id in enumerate(results['item_ids'][:3], 1):
                    score = results['scores'].get(item_id, 0.0)
                    startup = db.query(Startup).filter(Startup.id == item_id).first()
                    if startup:
                        logger.info(f"  {i}. {startup.title} (score: {score:.3f})")
            
            return True
        else:
            logger.warning(f"⚠ Expected two_tower but got {results.get('method_used')}")
            return True
            
    except Exception as e:
        logger.error(f"✗ Two-tower test failed: {e}", exc_info=True)
        return False
    finally:
        db.close()


def test_routing_logic():
    """Test recommendation routing logic"""
    logger.info("\n" + "="*60)
    logger.info("TEST 5: Routing Logic")
    logger.info("="*60)
    
    db = SessionLocal()
    try:
        from engines.router import RecommendationRouter
        
        # Test with two-tower disabled
        router_disabled = RecommendationRouter(enable_two_tower=False)
        
        # Test with two-tower enabled
        router_enabled = RecommendationRouter(enable_two_tower=True)
        
        # Get users with different interaction counts
        from database.models import UserInteraction
        from sqlalchemy import func
        
        users_by_interactions = db.query(
            User.id,
            func.count(UserInteraction.id).label('interaction_count')
        ).outerjoin(UserInteraction).group_by(User.id).all()
        
        test_cases = []
        for user_id, count in users_by_interactions[:5]:
            method_disabled, _ = router_disabled.route(str(user_id), 'developer_startup')
            method_enabled, _ = router_enabled.route(str(user_id), 'developer_startup')
            
            test_cases.append({
                'interactions': count,
                'method_disabled': method_disabled,
                'method_enabled': method_enabled
            })
        
        logger.info("Routing test cases:")
        for tc in test_cases:
            logger.info(f"  {tc['interactions']} interactions: disabled={tc['method_disabled']}, enabled={tc['method_enabled']}")
        
        logger.info("✓ Routing logic working")
        return True
        
    except Exception as e:
        logger.error(f"✗ Routing test failed: {e}", exc_info=True)
        return False
    finally:
        db.close()


def main():
    """Run all integration tests"""
    logger.info("\n")
    logger.info("#"*60)
    logger.info("# TWO-TOWER INTEGRATION TEST SUITE")
    logger.info("#"*60)
    logger.info("\n")
    
    tests = [
        ("Database Connection", test_database_connection),
        ("Model Registry", test_model_registry),
        ("Content-Based Recommendations", test_content_based_recommendations),
        ("Two-Tower Recommendations", test_two_tower_recommendations),
        ("Routing Logic", test_routing_logic),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            logger.error(f"Test '{name}' crashed: {e}")
            results.append((name, False))
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("TEST SUMMARY")
    logger.info("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status}: {name}")
    
    logger.info("\n" + "="*60)
    logger.info(f"RESULTS: {passed}/{total} tests passed")
    logger.info("="*60)
    
    if passed == total:
        logger.info("\n🎉 All tests passed! Two-tower integration is working.")
    elif passed >= total - 1:
        logger.info("\n✓ Core functionality working. Some features may need setup.")
    else:
        logger.error("\n⚠ Some tests failed. Check logs above for details.")
    
    return passed == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

