# Testing Quick Reference Card

## 🚀 Start Services

```bash
# Terminal 1 - Django
cd backend
python manage.py runserver

# Terminal 2 - Flask
cd recommendation_service
python app.py

# Terminal 3 - Run Tests
python test_complete_recommendation_flow.py
```

**OR** use automated script (Windows):
```bash
start_all_services.bat
```

---

## ✅ Health Checks

```bash
# Flask Health
curl http://localhost:5000/health

# Django Health  
curl http://localhost:8000/api/
```

---

## 🧪 Test Endpoints

### Direct Flask (No Auth Required)

```bash
# Developer Recommendations
curl "http://localhost:5000/api/recommendations/startups/for-developer/USER_ID?limit=5"

# Investor Recommendations
curl "http://localhost:5000/api/recommendations/startups/for-investor/USER_ID?limit=5"

# Trending (Public)
curl "http://localhost:5000/api/recommendations/trending/startups?limit=10"
```

### Django Proxy (Auth Required)

```bash
# Personalized Startups
curl -H "Authorization: Bearer TOKEN" \
     "http://localhost:8000/api/recommendations/personalized/startups?limit=5"
```

---

## 🎯 Expected Response

```json
{
  "startups": [...],
  "total": 10,
  "method_used": "als",           // Model used
  "interaction_count": 12,         // User's interaction history
  "model_version": "als_v1.0"
}
```

---

## 🔄 Routing Logic

| Interactions | Model | Method Used |
|--------------|-------|-------------|
| < 5 | Content-Based | `"content_based"` |
| 5-19 | ALS | `"als"` |
| 20+ | Ensemble | `"ensemble"` |

---

## 🛠️ Troubleshooting

### Services Not Running

```bash
# Kill stuck processes
netstat -ano | findstr :8000  # Django
netstat -ano | findstr :5000  # Flask
taskkill /PID <PID> /F

# Restart services
cd backend && python manage.py runserver
cd recommendation_service && python app.py
```

### Models Not Loading

```bash
# Train models
train_all_models.bat

# Restart Flask
cd recommendation_service && python app.py
```

---

## 📊 Success Criteria

✅ Django running (port 8000)  
✅ Flask running (port 5000)  
✅ Health endpoint returns healthy  
✅ Models loaded (check health response)  
✅ Recommendations return successfully  
✅ Automated tests pass (6/6)

---

## 📁 Key Files

| File | Purpose |
|------|---------|
| `test_complete_recommendation_flow.py` | Automated tests |
| `start_all_services.bat` | Auto-start script |
| `TESTING_SUMMARY.md` | Full testing guide |
| `MANUAL_TEST_GUIDE.md` | Manual testing procedures |

---

## 🎉 Quick Test (30 seconds)

```bash
# 1. Start services (2 terminals)
cd backend && python manage.py runserver
cd recommendation_service && python app.py

# 2. Run automated test (new terminal)
python test_complete_recommendation_flow.py

# Expected: "SUCCESS! ALL TESTS PASSED!"
```

---

## ❓ Need Help?

1. Check service logs
2. Verify database has data
3. Ensure models are trained
4. Review TESTING_SUMMARY.md
5. Follow MANUAL_TEST_GUIDE.md

---

## 🔗 Testing Flow

```
Frontend → Django → Flask → Router → Model → Response
```

Test each component individually, then test complete flow!

