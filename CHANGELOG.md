## 2025-09-29

### Fixed
- Restored collaboration application submissions by routing optional resume uploads through the resume endpoint and sending clean JSON payloads (`frontend/src/pages/ApplyJob/ApplyJob.jsx`).
- Surfaced recent application activity for students on the dashboard with status-aware styling (`frontend/src/pages/Dashboard/Dashboard.jsx`, `frontend/src/pages/Dashboard/Dashboard.module.css`).
- Repaired messaging flows so new conversations always include the sender and return full payloads for both conversation and message creation (`backend/api/serializers.py`, `backend/api/views.py`, `backend/api/tests.py`).
- Hardened messaging authentication by returning 401 for unauthenticated list requests and covering the full login → conversation → message round trip in automated tests (`backend/api/views.py`, `backend/api/tests.py`).

### Notes
- Frontend linting still reports longstanding warnings/errors in unrelated components; see `npm run lint` output for details.
