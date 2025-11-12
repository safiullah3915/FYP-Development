# Flask Recommendation Service

A scalable Flask-based recommendation service for the startup marketplace platform. This service connects to the Django SQLite database and provides a foundation for implementing recommendation algorithms.

## Overview

This service is designed to:
- Connect to the existing Django SQLite database (`backend/db.sqlite3`)
- Provide a clean, maintainable architecture for recommendation algorithms
- Support future expansion with services and API layers
- Scale to handle recommendation requests efficiently

## Project Structure

```
recommendation_service/
├── app.py                  # Flask application (minimal for now)
├── config.py               # Configuration management
├── requirements.txt        # Python dependencies
├── .env.example           # Environment variables template
├── README.md              # This file
│
├── database/
│   ├── __init__.py
│   ├── connection.py      # Database connection setup
│   └── models.py          # SQLAlchemy models (all Django tables)
│
└── utils/
    ├── __init__.py
    ├── logger.py          # Logging configuration
    └── helpers.py         # Utility functions
```

## Setup

### 1. Install Dependencies

```bash
cd recommendation_service
pip install -r requirements.txt
```

### 2. Configure Environment

Copy the example environment file and adjust if needed:

```bash
cp .env.example .env
```

The default configuration should work out of the box. The service will automatically connect to `backend/db.sqlite3`.

### 3. Run the Service

```bash
python app.py
```

The service will start on `http://localhost:5000` by default.

## API Endpoints

### Health Check

```bash
GET /health
```

Returns service status and database connection status.

**Response:**
```json
{
  "status": "healthy",
  "service": "recommendation-service",
  "version": "0.1.0",
  "database_connected": true
}
```

### Test Endpoints

These endpoints are for testing database connectivity:

```bash
GET /test/users      # Returns first 5 users
GET /test/startups   # Returns first 5 startups
```

## Database Connection

The service connects to the Django SQLite database at `backend/db.sqlite3`. All Django tables are available through SQLAlchemy models in `database/models.py`.

### Available Models

- **User** - User accounts
- **UserProfile** - Extended user profiles
- **UserOnboardingPreferences** - Onboarding preferences
- **Startup** - Startup listings
- **StartupTag** - Startup tags
- **Position** - Available positions
- **Application** - Job applications
- **Notification** - User notifications
- **Favorite** - User favorites
- **Interest** - Investor interests
- **Conversation** - User conversations
- **Message** - Messages
- **FileUpload** - File uploads
- **UserInteraction** - User interactions (for recommendations)
- **StartupTrendingMetrics** - Trending metrics
- **RecommendationModel** - ML model metadata

## Usage Example

```python
from database.connection import SessionLocal
from database.models import User, Startup

# Get database session
db = SessionLocal()

# Query users
users = db.query(User).filter(User.role == 'student').limit(10).all()

# Query startups
startups = db.query(Startup).filter(Startup.status == 'active').all()

# Don't forget to close the session
db.close()
```

## Configuration

Configuration is managed through environment variables (see `.env.example`):

- `FLASK_HOST` - Host to bind to (default: 0.0.0.0)
- `FLASK_PORT` - Port to bind to (default: 5000)
- `FLASK_DEBUG` - Enable debug mode (default: True)
- `LOG_LEVEL` - Logging level (default: INFO)
- `CORS_ORIGINS` - Allowed CORS origins (comma-separated)

## Logging

Logs are written to:
- Console (INFO level and above)
- File: `logs/recommendation_service.log` (DEBUG level and above)

## Future Expansion

This foundation is ready for:

1. **Services Layer** - Add recommendation algorithms:
   - Content-based filtering
   - Collaborative filtering
   - Deep learning models

2. **API Layer** - Add RESTful endpoints:
   - `/api/recommendations/startups/for-developer/<user_id>`
   - `/api/recommendations/developers/for-startup/<startup_id>`
   - `/api/recommendations/investors/for-startup/<startup_id>`

3. **Advanced Features**:
   - Caching layer (Redis)
   - Async support
   - Performance monitoring
   - Model versioning

## Development

### Running in Development

```bash
# Set debug mode
export FLASK_DEBUG=True

# Run the service
python app.py
```

### Testing Database Connection

```bash
# Health check
curl http://localhost:5000/health

# Test users query
curl http://localhost:5000/test/users

# Test startups query
curl http://localhost:5000/test/startups
```

## Notes

- The service reads from the Django database but does not modify it
- SQLite connection pooling is configured for concurrent access
- All models match Django schema exactly
- JSON fields are stored as Text in SQLite and parsed when accessed

## Troubleshooting

### Database Connection Issues

If you see database connection errors:

1. Verify `backend/db.sqlite3` exists
2. Check file permissions
3. Ensure Django migrations have been run

### Import Errors

If you see import errors:

1. Ensure you're in the `recommendation_service` directory
2. Verify all dependencies are installed: `pip install -r requirements.txt`
3. Check Python path includes the project root

## License

Part of the FYP Development project.

