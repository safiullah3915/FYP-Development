# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

This is a full-stack startup marketplace platform with separate Django backend and React frontend. The platform enables entrepreneurs to list startups for sale or collaboration, students to find positions, and investors to discover opportunities.

**Backend**: Django REST API with JWT authentication, 34+ endpoints
**Frontend**: React + Vite with role-based authentication and navigation

## Common Development Commands

### Backend Commands (Django)

```powershell
# Navigate to backend directory
cd backend

# Create and activate virtual environment (Windows)
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Environment setup
copy .env.example .env
# Edit .env with your configuration

# Database operations
python manage.py makemigrations
python manage.py migrate
python manage.py createsuperuser

# Run development server
python manage.py runserver
# API available at: http://localhost:8000

# Run tests
python manage.py test
python manage.py test api.tests.TestAuthentication
python manage.py test api.tests.TestStartups

# Test coverage
pip install coverage
coverage run --source='.' manage.py test
coverage report

# Collect static files (production)
python manage.py collectstatic --noinput
```

### Frontend Commands (React + Vite)

```powershell
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev
# Available at: http://localhost:5174

# Build for production
npm run build

# Lint code
npm run lint

# Preview production build
npm run preview

# Deploy to GitHub Pages
npm run deploy
```

## Architecture Overview

### Backend Structure (Django)

- **API App**: Single Django app containing all business logic
- **Models**: User (3 roles), Startup, Position, Application, Notification, Favorite, Interest
- **Authentication**: JWT with HTTP-only cookies, email verification system
- **Role-based Access**: Entrepreneur, Student/Professional, Investor roles

**Key Backend Patterns**:
- UUID primary keys for enhanced security
- JSON fields for flexible data storage (stages, startup data)
- Comprehensive indexing for query performance
- Status fields for workflow management (Application.status, Startup.status)
- Related models with proper foreign key relationships

### Frontend Structure (React)

- **Role-based Routing**: Different UI components and access levels per user role
- **Context-based Auth**: `AuthContext.jsx` provides authentication state
- **Protected Routes**: `ProtectedRoute.jsx` and `RoleBasedRoute.jsx` components
- **Modular Components**: Organized by feature (Navbar, Cards, Forms)

**Key Frontend Patterns**:
- Role-specific navigation bars
- Cookie-based authentication with backend
- Form validation and error handling
- Responsive design with Tailwind CSS

### Database Models

**Core Entities**:
- `User`: Extended Django user with roles (entrepreneur/student/investor)
- `Startup`: Central entity supporting both marketplace and collaboration types
- `Position`: Available roles in collaboration startups
- `Application`: Job applications with status tracking

**Supporting Models**:
- `Notification`: In-app notification system
- `Favorite`: Investor engagement tracking
- `Interest`: Expression of investment interest
- `StartupTag`: Flexible tagging system

## Role-Based Features

### Entrepreneurs
- Create marketplace listings (startups for sale)
- Create collaboration listings (looking for team)
- Manage applications from students
- Pitch business ideas to investors

### Students/Professionals
- Browse collaboration opportunities
- Apply for positions with cover letters
- Upload resumes and portfolios
- Track application status

### Investors
- Browse marketplace startups
- Express interest in startups
- Favorite promising opportunities
- Access investor-specific dashboard

## Authentication Flow

1. **Registration**: Email/password with role selection
2. **Email Verification**: 6-digit code system (check backend logs for codes during development)
3. **Login**: JWT tokens stored in HTTP-only cookies
4. **Access Control**: Role-based route protection

## Development Environment

### Ports
- Backend (Django): `http://localhost:8000`
- Frontend (Vite): `http://localhost:5174`

### Database Configuration
- **Development**: SQLite (default)
- **Production Options**: PostgreSQL or MySQL
- Migrations are version-controlled and should be applied after model changes

### CORS Configuration
- Backend configured for `http://localhost:3000` and `http://127.0.0.1:3000`
- Update `CORS_ALLOWED_ORIGINS` in `.env` for different frontend ports

## Testing Strategy

### Backend Testing
- Unit tests for models, views, and authentication
- API endpoint testing with Django's test client
- Authentication flow testing scripts available

### Frontend Testing  
- Manual testing guide in `frontend/TESTING_GUIDE.md`
- Role-based navigation testing
- Form submission and error handling tests

### Integration Testing
- Full authentication flow testing
- Cross-role interaction testing (entrepreneur → student applications)
- API endpoint testing with actual frontend requests

## Common Issues and Solutions

### Authentication Issues
- **Token Errors**: Check JWT configuration in backend `.env`
- **CORS Errors**: Verify `CORS_ALLOWED_ORIGINS` includes frontend URL
- **Login Redirect**: Ensure role-based routing is properly configured

### Database Issues
- **Migration Errors**: Delete migration files and recreate with `makemigrations`
- **Foreign Key Errors**: Check model relationships and related_name conflicts
- **UUID Issues**: Ensure UUID fields are properly handled in serializers

### Development Server Issues
- **Port Conflicts**: Backend uses 8000, frontend uses 5174 by default
- **Environment Variables**: Copy `.env.example` to `.env` in backend
- **Static Files**: Run `collectstatic` if CSS/JS not loading

## File Upload Handling

The platform supports multiple file types:
- Resume uploads for applications
- Profile pictures for users  
- Startup images for listings
- Message attachments

**Upload Endpoints**:
- `/api/upload/resume`
- `/api/upload/profile-picture`
- `/api/upload/startup-image`

## API Integration Patterns

### Frontend → Backend Communication
- Credentials: `include` for cookie-based authentication
- Content-Type: `application/json` for most requests
- Error Handling: Check response status and display user-friendly messages

### Sample API Call Pattern:
```javascript
const response = await fetch('http://localhost:8000/api/endpoint', {
  method: 'POST',
  credentials: 'include',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(data)
});
```

## Security Considerations

- JWT tokens in HTTP-only cookies (not localStorage)
- Rate limiting on authentication endpoints
- Input validation on both frontend and backend
- UUID primary keys to prevent enumeration attacks
- CSRF protection enabled
- Email verification required for activation

## Production Deployment Notes

### Backend Checklist
- Set `DEBUG=False`
- Configure production database (PostgreSQL recommended)
- Set `JWT_COOKIE_SECURE=True` for HTTPS
- Configure email backend for verification emails
- Set up proper static file serving
- Configure logging and monitoring

### Frontend Checklist  
- Update API URLs for production backend
- Configure proper build process
- Set up CDN for static assets
- Update CORS configuration in backend

## Code Organization Principles

- **Backend**: Single `api` app with feature-based organization
- **Frontend**: Component-based architecture with shared contexts
- **Separation of Concerns**: Authentication, business logic, and presentation clearly separated
- **Consistent Naming**: RESTful API endpoints, descriptive component names
- **Error Handling**: Comprehensive error responses with user-friendly messages
