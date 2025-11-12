# Startup Sales & Purchase Platform - Backend API

🚀 **A comprehensive Django REST API for startup sales and collaboration platform with 34+ endpoints ready for production**

A feature-rich Django backend that enables entrepreneurs to list their startups for sale, find collaboration opportunities, apply for positions, and engage with investors through a complete marketplace ecosystem.

---

## ✨ Key Features

### 🔐 **Authentication & Security**
- JWT-based authentication with HTTP-only cookies
- Email verification system with 6-digit codes
- Rate limiting on authentication endpoints
- bcrypt password hashing with configurable rounds
- CSRF protection and CORS configuration
- Session management with access/refresh token pairs

### 🏢 **Startup Management**
- Create marketplace and collaboration listings
- Advanced search and filtering capabilities
- Startup categorization (SaaS, E-commerce, Agency, etc.)
- Performance metrics tracking (TTM revenue, profit)
- Tags and field-based organization
- View tracking and featured listings

### 📋 **Application System**
- Apply for startup positions with cover letters
- Application status tracking (pending, approved, rejected)
- Position management for entrepreneurs
- Bulk application handling
- Portfolio URL and experience tracking

### 💬 **Messaging System**
- Real-time conversations between users
- File attachments (images, documents)
- Online user status tracking
- Conversation management and history
- Message read status tracking

### 👤 **User Profiles**
- Extended profile management
- Skills, experience, and reference tracking
- Profile picture uploads
- Public/private profile settings
- Regional preferences and location data

### 📁 **File Management**
- Resume uploads for applications
- Startup image uploads
- Profile picture management
- Message attachments
- File type validation and storage

### 🔔 **Notifications**
- Real-time notification system
- Application status updates
- Message notifications
- Bulk read/unread management

### ⭐ **Investor Features**
- Favorite startup listings
- Express interest in startups
- Investment tracking and management
- Investor-specific dashboard data

### 📊 **Analytics & Search**
- Platform statistics and metrics
- Advanced search with filters
- User activity tracking
- Performance analytics

---

## 🛠️ Tech Stack

| Technology | Version | Purpose |
|------------|---------|----------|
| **Django** | Latest | Web framework |
| **Django REST Framework** | Latest | API framework |
| **JWT** | PyJWT | Authentication tokens |
| **bcrypt** | Latest | Password hashing |
| **MySQL/PostgreSQL/SQLite** | - | Database options |
| **CORS Headers** | django-cors-headers | Cross-origin requests |
| **Pillow** | Latest | Image processing |
| **Rate Limiting** | django-ratelimit | API rate limiting |
| **Python Decouple** | Latest | Environment management |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip package manager
- Virtual environment (recommended)

### Installation

1. **Clone and navigate to backend**
   ```bash
   git clone <repository-url>
   cd backend
   ```

2. **Create and activate virtual environment**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate
   
   # Linux/Mac
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment configuration**
   ```bash
   copy .env.example .env  # Windows
   cp .env.example .env    # Linux/Mac
   # Edit .env with your configuration
   ```

5. **Database setup**
   ```bash
   python manage.py makemigrations
   python manage.py migrate
   python manage.py createsuperuser
   ```

6. **Start the development server**
   ```bash
   python manage.py runserver
   ```

✅ **API available at:** `http://localhost:8000`

---

## 🔗 API Endpoints Overview

> **Total Endpoints: 34+** | **Status: Production Ready** ✅

### 🔐 Authentication (7 endpoints)
```
POST   /signup                         # User registration
POST   /auth/login                     # User login
POST   /auth/refresh                   # Refresh tokens
POST   /auth/logout                    # User logout
POST   /auth/forget-password           # Password reset
POST   /auth/send-verification-code    # Send email verification
POST   /auth/verify                    # Verify email
```

### 🏢 Startup Management (4 endpoints)
```
POST   /api/startups                   # Create startup
GET    /api/startups/{id}              # Get startup details
GET    /api/marketplace                # List marketplace startups
GET    /api/collaborations             # List collaboration startups
```

### 📋 Applications (6 endpoints)
```
POST   /api/collaborations/{id}/apply  # Apply for position
GET    /api/users/applications         # Get user applications
GET    /api/startups/{id}/applications # Get startup applications
POST   /api/applications/{id}/approve  # Approve application
POST   /api/applications/{id}/decline  # Decline application
GET    /api/positions                  # List all positions
```

### 💬 Messaging (4 endpoints)
```
GET    /api/messages                   # List conversations
GET    /api/messages/{id}              # Get conversation details
GET    /api/messages/{id}/messages     # Get conversation messages
GET    /api/messages/users/online      # Get online users
```

### 👤 User Management (4 endpoints)
```
GET    /api/users/profile              # Get user profile
PUT    /api/users/profile              # Update user profile
GET    /api/users/profile-data         # Get comprehensive profile data
GET    /account/{token}                # Get account by token
```

### 📁 File Uploads (5 endpoints)
```
POST   /api/upload                     # General file upload
POST   /api/upload/resume              # Upload resume
POST   /api/upload/startup-image       # Upload startup image
POST   /api/upload/profile-picture     # Upload profile picture
GET    /api/uploads                    # List user uploads
```

### 🔔 Notifications (3 endpoints)
```
GET    /api/notifications              # List notifications
POST   /api/notifications/{id}/read    # Mark notification read
POST   /api/notifications/read-all     # Mark all notifications read
```

### ⭐ Investor Features (4 endpoints)
```
GET    /api/users/favorites            # List user favorites
POST   /api/startups/{id}/favorite     # Toggle favorite
GET    /api/users/interests            # List user interests
POST   /api/startups/{id}/interest     # Express interest
```

### 📊 Analytics (2 endpoints)
```
GET    /api/stats                      # Platform statistics
GET    /api/search                     # Search startups
```


---

## 🗄️ Database Models

### Core Models
- **User** - Extended Django user with roles (entrepreneur, student, investor)
- **Startup** - Main entity for marketplace/collaboration listings
- **StartupTag** - Tagging system for startups
- **Position** - Available positions in startups
- **Application** - Job applications with status tracking
- **EmailVerificationCode** - Email verification system

### Extended Models
- **Conversation** - User messaging conversations
- **Message** - Individual messages with attachments
- **UserProfile** - Extended user information and settings
- **FileUpload** - File management system
- **Notification** - Real-time notification system
- **Favorite** - User favorite startups
- **Interest** - Investor interest expressions

### Model Features
- **UUID Primary Keys** - Enhanced security
- **JSON Fields** - Flexible data storage
- **Database Indexing** - Optimized queries
- **Timestamp Tracking** - Created/updated timestamps
- **Status Management** - Workflow state tracking

---

## ⚙️ Configuration

### 🔧 Environment Variables

```env
# Django Settings
SECRET_KEY=your-secret-key-here
DEBUG=True
ALLOWED_HOSTS=localhost,127.0.0.1

# Database Configuration (SQLite default)
DB_ENGINE=django.db.backends.sqlite3
DB_NAME=db.sqlite3

# JWT Configuration
JWT_SECRET_KEY=your-jwt-secret-key-here
JWT_COOKIE_SECURE=False  # True for production

# CORS Configuration
CORS_ALLOWED_ORIGINS=http://localhost:3000,http://127.0.0.1:3000

# Email Configuration
EMAIL_BACKEND=django.core.mail.backends.console.EmailBackend
EMAIL_HOST=smtp.gmail.com
EMAIL_PORT=587
```

### 🗄️ Database Options

**MySQL Configuration**
```env
DB_ENGINE=django.db.backends.mysql
DB_NAME=startup_platform
DB_USER=your_db_user
DB_PASSWORD=your_db_password
DB_HOST=localhost
DB_PORT=3306
```

**PostgreSQL Configuration**
```env
DB_ENGINE=django.db.backends.postgresql
DB_NAME=startup_platform
DB_USER=your_db_user
DB_PASSWORD=your_db_password
DB_HOST=localhost
DB_PORT=5432
```

---

## 🔒 Security Features

### Authentication Security
- ✅ JWT tokens in HTTP-only cookies
- ✅ Access/refresh token architecture
- ✅ Email verification requirement
- ✅ Rate limiting (5 attempts/minute for signup, 10 for login)
- ✅ bcrypt password hashing
- ✅ Session invalidation on logout

### API Security
- ✅ CORS configuration
- ✅ CSRF protection
- ✅ Input validation and sanitization
- ✅ SQL injection protection
- ✅ File upload validation
- ✅ Permission-based access control

### Data Security
- ✅ UUID primary keys
- ✅ Sensitive data encryption
- ✅ Secure cookie configuration
- ✅ Environment variable management

---

## 🧪 Testing

### Run Test Suite
```bash
python manage.py test
```

### Run Specific Tests
```bash
python manage.py test api.tests.TestAuthentication
python manage.py test api.tests.TestStartups
```

### Test Coverage
```bash
pip install coverage
coverage run --source='.' manage.py test
coverage report
```

---

## 📊 Performance Features

### Database Optimization
- **Indexed Fields** - Optimized query performance
- **Select Related** - Reduced database queries
- **Pagination** - Efficient data loading
- **Query Optimization** - Minimal N+1 queries

### Caching (Ready for Implementation)
- Redis cache configuration ready
- Session caching supported
- Query result caching prepared

### File Handling
- **Efficient Uploads** - Chunked file processing
- **Media Serving** - Optimized static file serving
- **File Validation** - Type and size restrictions

---

## 🚀 Production Deployment

### Pre-deployment Checklist
- [ ] Set `DEBUG=False`
- [ ] Configure production database (PostgreSQL/MySQL)
- [ ] Set `JWT_COOKIE_SECURE=True`
- [ ] Configure proper CORS origins
- [ ] Set up SSL/HTTPS
- [ ] Configure email backend
- [ ] Set up logging
- [ ] Configure static file serving
- [ ] Set up monitoring
- [ ] Configure backups

### Deployment Commands
```bash
# Collect static files
python manage.py collectstatic --noinput

# Run migrations
python manage.py migrate

# Create superuser
python manage.py createsuperuser
```

### Recommended Production Stack
- **Web Server**: Nginx
- **WSGI Server**: Gunicorn
- **Database**: PostgreSQL
- **Cache**: Redis
- **File Storage**: AWS S3 or similar
- **Monitoring**: Sentry

---

## 📈 API Response Formats

### Success Response
```json
{
  "message": "Success message",
  "data": {...},
  "pagination": {
    "currentPage": 1,
    "totalPages": 5,
    "totalItems": 60,
    "itemsPerPage": 12
  }
}
```

### Error Response
```json
{
  "error": "Error message",
  "detail": "Detailed error information"
}
```

### HTTP Status Codes
- `200` - Success
- `201` - Created
- `400` - Bad Request
- `401` - Unauthorized
- `403` - Forbidden
- `404` - Not Found
- `409` - Conflict
- `429` - Too Many Requests
- `500` - Internal Server Error

---

## 🎯 Frontend Integration

### Ready for Integration
✅ **React/Vue/Angular** compatible
✅ **Mobile app** ready (React Native/Flutter)
✅ **Cookie-based authentication** for web
✅ **Token-based authentication** for mobile
✅ **CORS configured** for all common ports
✅ **File upload** endpoints ready
✅ **Real-time features** prepared

### Sample Frontend Integration
```javascript
// Login example
const response = await fetch('http://localhost:8000/auth/login', {
  method: 'POST',
  credentials: 'include', // Important for cookies
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    email: 'user@example.com',
    password: 'password123'
  })
});

const data = await response.json();
if (data.message === 'Login successful') {
  // User authenticated, cookies set automatically
  console.log('User:', data.user);
}
```

---

## 📝 Development Guidelines

### Code Style
- Follow PEP 8 style guide
- Use meaningful variable names
- Add docstrings to functions/classes
- Keep functions small and focused

### Git Workflow
1. Create feature branch from `main`
2. Make changes with clear commit messages
3. Test thoroughly before push
4. Create pull request for review
5. Merge after approval

### Adding New Features
1. Create model in `models.py`
2. Add serializer in `serializers.py`
3. Create view in `views.py`
4. Add URL pattern in `urls.py`
5. Write tests in `tests.py`
6. Update API documentation

---

## 🤝 Contributing

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Contribution Guidelines
- Write clear commit messages
- Add tests for new features
- Update documentation
- Follow existing code style
- Test your changes thoroughly

---

## 📞 Support & Documentation

- 📚 **API Documentation**: [API_DOCUMENTATION.md](./API_DOCUMENTATION.md)
- 🐛 **Issue Tracking**: GitHub Issues
- 💬 **Discussions**: GitHub Discussions
- 📧 **Contact**: [Your contact information]

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🎉 Project Status

**🟢 Production Ready** | **34+ Endpoints** | **Full CRUD Operations** | **Secure Authentication** | **File Uploads** | **Real-time Features**

> **Ready for frontend integration!** All endpoints are implemented, tested, and documented. The backend provides a complete foundation for building a modern startup marketplace platform.

---

*Last Updated: September 2024*
