# How to Start All Services for Testing

## Quick Start (3 Steps)

### Step 1: Start Django Backend

Open a terminal and run:

```bash
cd backend
python manage.py runserver
```

**Expected output:**
```
Starting development server at http://127.0.0.1:8000/
Quit the server with CTRL-BREAK.
```

Leave this terminal open.

---

### Step 2: Start Flask Recommendation Service

Open a **NEW** terminal and run:

```bash
cd recommendation_service
python app.py
```

**Expected output:**
```
[OK] Two-Tower model loaded successfully!
[OK] ALS model loaded successfully!
[OK] Ensemble model initialized successfully!
  -> Routing: cold start(<5) -> content, warm(5-19) -> ALS, hot(20+) -> ensemble
 * Running on http://127.0.0.1:5000
```

Leave this terminal open.

---

### Step 3: Run Tests

Open a **THIRD** terminal and run:

```bash
python test_complete_recommendation_flow.py
```

This will test the complete flow:
- Django health check
- Flask health check
- Direct Flask endpoints
- Django proxy endpoints
- Smart routing logic
- Response quality

---

## What Each Service Does

### Django Backend (Port 8000)
- Main application API
- User authentication
- Database management
- **Proxy endpoints** that forward to Flask

### Flask Recommendation Service (Port 5000)
- ML models (ALS, Two-Tower, Ensemble)
- Recommendation generation
- Smart routing logic

### Test Flow
```
Frontend
   |
   v
Django (8000) --proxy--> Flask (5000) --> Models (ALS/Two-Tower/Ensemble)
   ^                                            |
   |                                            |
   +--------------------Response----------------+
```

---

## Troubleshooting

### Port Already in Use

**Django (8000):**
```bash
# Find process
netstat -ano | findstr :8000

# Kill process (replace PID)
taskkill /PID <PID> /F

# Restart Django
cd backend && python manage.py runserver
```

**Flask (5000):**
```bash
# Find process
netstat -ano | findstr :5000

# Kill process (replace PID)
taskkill /PID <PID> /F

# Restart Flask
cd recommendation_service && python app.py
```

### Models Not Loading

If Flask shows warnings about models not loading:

```bash
# Train models first
./train_all_models.bat

# Then restart Flask
cd recommendation_service
python app.py
```

### Database Issues

If Django shows database errors:

```bash
cd backend
python manage.py migrate
python manage.py runserver
```

---

## Full Test Checklist

- [ ] Django running on port 8000
- [ ] Flask running on port 5000
- [ ] Flask loaded ALS model
- [ ] Flask loaded Two-Tower model
- [ ] Flask initialized Ensemble model
- [ ] Test script passes all tests

---

## Quick Commands Reference

| Action | Command |
|--------|---------|
| Start Django | `cd backend && python manage.py runserver` |
| Start Flask | `cd recommendation_service && python app.py` |
| Run Tests | `python test_complete_recommendation_flow.py` |
| Train Models | `train_all_models.bat` |
| Check Django | `curl http://localhost:8000/api/` |
| Check Flask | `curl http://localhost:5000/health` |

---

## Next: Run Tests

Once both services are running, execute:

```bash
python test_complete_recommendation_flow.py
```

You should see:
```
SUCCESS! ALL TESTS PASSED! System is working correctly.
```

