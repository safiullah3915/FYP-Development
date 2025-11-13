# Environment Setup Guide for Running Migrations

## Quick Setup Steps

### 1. Activate the Virtual Environment

**For PowerShell (Windows):**
```powershell
cd backend
.\env\Scripts\Activate.ps1
```

**For Command Prompt (Windows):**
```cmd
cd backend
env\Scripts\activate.bat
```

**For Git Bash / Linux / Mac:**
```bash
cd backend
source env/bin/activate
```

### 2. Install/Update Dependencies

Once the virtual environment is activated (you should see `(env)` in your terminal), install the required packages:

```powershell
pip install -r requirements.txt
```

If you encounter permission issues, you might need to run:
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Verify Installation

Check that Django and django-apscheduler are installed:
```powershell
python -c "import django; import django_apscheduler; print('Dependencies OK')"
```

### 4. Run Migrations

Now you can generate and apply the migrations:

```powershell
# Generate migration for the new RecommendationSession model
python manage.py makemigrations

# Apply the migration to the database
python manage.py migrate
```

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'django_apscheduler'"

**Solution:** Make sure you've activated the virtual environment and installed requirements:
```powershell
.\env\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Issue: "Execution Policy" error in PowerShell

If you get an execution policy error when trying to activate the virtual environment:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then try activating again:
```powershell
.\env\Scripts\Activate.ps1
```

### Issue: Virtual environment not found

If the `env` folder doesn't exist or is corrupted, create a new one:

```powershell
# Remove old env if it exists
Remove-Item -Recurse -Force env

# Create new virtual environment
python -m venv env

# Activate it
.\env\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### Issue: Database connection errors

Make sure your database settings in `.env` file (or environment variables) are correct. For SQLite (default), no configuration is needed.

## What the Migration Will Do

The migration will:
1. Create the `recommendation_sessions` table with all the ETL-optimized fields
2. Add the new index to the `user_interactions` table for ETL queries
3. Set up all the necessary database constraints and indexes

## After Migration

Once migrations are complete, you can:
- Start using the recommendation session tracking
- Test the new endpoints
- Verify that sessions are being stored correctly

## Quick Command Reference

```powershell
# Activate environment
.\env\Scripts\Activate.ps1

# Install/update dependencies
pip install -r requirements.txt

# Generate migrations
python manage.py makemigrations

# Apply migrations
python manage.py migrate

# Check migration status
python manage.py showmigrations

# Deactivate environment (when done)
deactivate
```

