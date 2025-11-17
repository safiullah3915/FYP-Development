# Reverse ALS Implementation - Complete Summary

## 🎯 Problem Solved

**Original Issue:**
- ALS only worked for User → Startup recommendations
- Founders couldn't get personalized Developer/Investor recommendations
- Only content-based matching was available for founder use cases

**Solution Implemented:**
- Created bi-directional ALS system using transposed interaction matrix
- No new tables needed - reuses existing `UserInteraction` data
- Full collaborative filtering for ALL use cases

## 📊 Use Case Coverage

| Use Case | Before | After | Model Used |
|----------|--------|-------|------------|
| Developer → Startup | ✅ ALS | ✅ ALS | `als_v1.pkl` |
| Investor → Startup | ✅ ALS | ✅ ALS | `als_v1.pkl` |
| Founder → Developer | ❌ Content-only | ✅ ALS Reverse | `als_reverse_v1.pkl` |
| Founder → Investor | ❌ Content-only | ✅ ALS Reverse | `als_reverse_v1.pkl` |
| Entrepreneur → Startup (Collab) | ✅ ALS | ✅ ALS | `als_v1.pkl` |

## 🔧 Files Created

1. **`recommendation_service/train_als_reverse.py`**
   - Trains reverse ALS model on Startups × Users matrix
   - Same hyperparameters as forward model
   - Outputs: `als_reverse_v1.pkl` + factors + mappings

2. **`recommendation_service/inference_als_reverse.py`**
   - Inference wrapper for reverse recommendations
   - Recommends users to startups
   - Handles role filtering (student/investor)

3. **`recommendation_service/REVERSE_ALS_SETUP.md`**
   - Complete setup and testing guide
   - Troubleshooting tips
   - Performance benchmarks

## 📝 Files Modified

1. **`backend/api/management/commands/generate_als_dataset.py`**
   - Added `build_reverse_sparse_matrix()` method
   - Generates both forward and reverse matrices
   - Saves 6 files total (3 forward, 3 reverse)

2. **`recommendation_service/app.py`**
   - Loads both ALS models on startup
   - Updated `/api/recommendations/developers/for-startup/<startup_id>`
   - Updated `/api/recommendations/investors/for-startup/<startup_id>`
   - Added smart routing based on interaction count

3. **`recommendation_service/engines/content_based.py`**
   - Implemented `recommend_users_for_startup()` method
   - Added `_calculate_user_startup_similarity()` scoring
   - Added `_generate_user_match_reasons()` explanations

4. **`recommendation_service/engines/router.py`**
   - Added `route_reverse()` method
   - Added `_get_startup_interaction_count()` method
   - Counts unique users who interacted with startup

## 🚀 How It Works

### Data Flow

```
UserInteraction Table (user_id, startup_id, weight)
            ↓
    generate_als_dataset
            ↓
    ┌───────────────────┐
    │  Forward Matrix   │  Reverse Matrix
    │  (Users × Startups)│  (Startups × Users)
    └───────────────────┘
            ↓
    ┌───────────────────┐
    │  train_als.py     │  train_als_reverse.py
    └───────────────────┘
            ↓
    ┌───────────────────┐
    │  als_v1.pkl       │  als_reverse_v1.pkl
    └───────────────────┘
```

### Routing Logic

**Forward (User → Startup):**
- User has <5 interactions → Content-Based
- User has 5-19 interactions → ALS Forward
- User has 20+ interactions → Ensemble (ALS + Two-Tower)

**Reverse (Startup → User):**
- Startup has <5 unique user interactions → Content-Based
- Startup has ≥5 unique user interactions → ALS Reverse

### Matrix Transposition

**Original Matrix (Users × Startups):**
```
        Startup1  Startup2  Startup3
User1      5.0      0.0      3.5
User2      0.0      8.0      2.0
User3      1.5      0.0      0.0
```

**Transposed Matrix (Startups × Users):**
```
           User1  User2  User3
Startup1    5.0    0.0    1.5
Startup2    0.0    8.0    0.0
Startup3    3.5    2.0    0.0
```

**Same data, different perspective!**

## 📦 Model Files Structure

```
recommendation_service/models/
├── als_v1.pkl                          # Forward model
├── als_v1_user_factors.npy             # User embeddings (128-dim)
├── als_v1_item_factors.npy             # Startup embeddings (128-dim)
├── als_v1_user_mapping.json            # user_id → index
├── als_v1_item_mapping.json            # startup_id → index
│
├── als_reverse_v1.pkl                  # Reverse model
├── als_reverse_v1_user_factors.npy     # Startup embeddings (128-dim)
├── als_reverse_v1_item_factors.npy     # User embeddings (128-dim)
├── als_reverse_v1_user_mapping.json    # startup_id → index
├── als_reverse_v1_item_mapping.json    # user_id → index
```

## 🧪 Testing Commands

### 1. Generate Data
```bash
cd backend
python manage.py generate_als_dataset
```

### 2. Train Models
```bash
cd ../recommendation_service
python train_als.py
python train_als_reverse.py
```

### 3. Start Service
```bash
python app.py
```

### 4. Test Endpoints

**Developer → Startup:**
```bash
curl "http://localhost:5001/api/recommendations/startups/for-developer/<USER_ID>?limit=10"
```

**Founder → Developer:**
```bash
curl "http://localhost:5001/api/recommendations/developers/for-startup/<STARTUP_ID>?limit=10"
```

**Founder → Investor:**
```bash
curl "http://localhost:5001/api/recommendations/investors/for-startup/<STARTUP_ID>?limit=10"
```

## ✅ Success Criteria Met

- [x] ALS works for User → Startup (original functionality preserved)
- [x] ALS works for Startup → User (new functionality added)
- [x] No new database tables created
- [x] Reuses existing UserInteraction data
- [x] Automatic routing based on interaction counts
- [x] Graceful fallback to content-based for cold-start
- [x] All endpoints updated and tested
- [x] Documentation and setup guide created

## 🔄 Retraining Workflow

**When to retrain:**
- Weekly for active platforms
- After significant user growth (2x interactions)
- When recommendation quality decreases

**Quick retrain:**
```bash
cd backend && python manage.py generate_als_dataset
cd ../recommendation_service
python train_als.py && python train_als_reverse.py
# Restart Flask service - models auto-load
```

## 📈 Expected Performance

**Good Metrics:**
- Forward ALS Precision@10: > 0.15
- Reverse ALS Precision@10: > 0.12
- API Response Time: < 200ms
- Cold-start Coverage: 100% (via content-based fallback)

## 🎓 Key Insights

1. **Matrix Transposition is Powerful:** Same data, different perspective enables bi-directional recommendations
2. **No New Tracking Needed:** Existing interactions work both ways
3. **Graceful Degradation:** Content-based fallback ensures 100% coverage
4. **Automatic Routing:** System intelligently chooses best model based on data

## 🚀 Next Steps (Optional Improvements)

1. **Ensemble for Reverse:** Combine ALS Reverse + Content-Based for better results
2. **Position-Specific Matching:** Filter developers by specific position requirements
3. **Active Learning:** Prompt founders to rate recommended users
4. **Cross-Validation:** Evaluate reverse model quality more thoroughly
5. **Real-time Updates:** Incremental model updates without full retraining

## 📞 Support

- Setup Guide: `recommendation_service/REVERSE_ALS_SETUP.md`
- Training Guide: `recommendation_service/ALS_TRAINING_GUIDE.md`
- Logs: `recommendation_service/logs/recommendation_service.log`

---

**Implementation Complete! 🎉**

All use cases now have full collaborative filtering support. The system is production-ready and will automatically route to the appropriate model based on available interaction data.

