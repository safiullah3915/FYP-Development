# UUID Format Mismatch Fix

## Problem

The Flask recommendation service was showing `interaction_count=0` for users who actually had more than 5 interactions, causing the system to always use content-based recommendations instead of two-tower/ALS models.

## Root Cause

**UUID Format Mismatch:**
- Django stores UUIDs **with dashes**: `3ef204fd-d6b3-4a7e-a376-27758efc383d`
- SQLite (via SQLAlchemy) reads UUIDs **without dashes**: `3ef204fdd6b34a7ea37627758efc383d`
- The Flask service was querying with dashed UUIDs, but SQLite had stored them without dashes
- This caused all queries to return 0 results, making the system think users had no interactions

## Solution

Normalize UUID format by removing dashes before querying the database in:
1. `recommendation_service/engines/router.py` - `_get_interaction_count()` method
2. `recommendation_service/app.py` - Two places where interaction count is checked

### Code Changes

**Before:**
```python
count = db.query(UserInteraction).filter(
    UserInteraction.user_id == user_id
).count()
```

**After:**
```python
# Normalize UUID format (remove dashes) since SQLite stores UUIDs without dashes
normalized_user_id = str(user_id).replace('-', '')
count = db.query(UserInteraction).filter(
    UserInteraction.user_id == normalized_user_id
).count()
```

## Files Modified

1. `recommendation_service/engines/router.py`
   - Fixed `_get_interaction_count()` method

2. `recommendation_service/app.py`
   - Fixed interaction count check in `/api/recommendations/startups/for-developer/<user_id>` endpoint
   - Fixed interaction count check in `/api/recommendations/startups/for-investor/<user_id>` endpoint

## Testing

To verify the fix works:

1. **Check user interactions in Django:**
   ```bash
   cd backend
   python check_user_interactions.py
   ```

2. **Test the Flask service:**
   - Restart the Flask recommendation service
   - Make a request to `/api/recommendations/startups/for-developer/<user_id>`
   - Check logs - should now show correct interaction count
   - For users with >5 interactions, should route to two-tower/ALS instead of content-based

## Expected Behavior After Fix

- Users with **0-4 interactions**: Content-based recommendations (cold start)
- Users with **5-19 interactions**: Two-tower or ALS recommendations (warm users)
- Users with **20+ interactions**: Ensemble recommendations (hot users)

## Note

Other services (`interaction_service.py`, `filter_service.py`, `business_rules.py`) may also need UUID normalization if they directly query by user_id. However, since they're typically called after the router has already normalized the ID, they should work correctly. If issues persist, apply the same normalization pattern to those files.

