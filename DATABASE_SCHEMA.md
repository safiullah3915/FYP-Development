# Complete Database Schema Documentation

## Overview
This is a Django-based startup marketplace platform database schema. The system supports entrepreneurs, students/professionals, and investors with features for startup listings, job applications, messaging, and investor engagement.

---

## Table of Contents
1. [Core Tables](#core-tables)
2. [Startup & Job System](#startup--job-system)
3. [User Engagement](#user-engagement)
4. [Messaging System](#messaging-system)
5. [Key Relationships Explained](#key-relationships-explained)

---

## Core Tables

### 1. `users` (User)
**Purpose:** Core authentication and user account information. Extends Django's AbstractUser.

**Columns:**
- `id` (UUID, Primary Key) - Auto-generated UUID
- `email` (EmailField, UNIQUE, REQUIRED) - Used as USERNAME_FIELD for login
- `username` (CharField, max_length=150, UNIQUE, REQUIRED) - Django username field
- `password` (CharField, max_length=128) - Hashed password (Django default)
- `first_name` (CharField, max_length=150, blank=True)
- `last_name` (CharField, max_length=150, blank=True)
- `is_superuser` (BooleanField, default=False) - Admin access
- `is_staff` (BooleanField, default=False) - Staff access
- `is_active` (BooleanField, default=True) - Account active status
- `email_verified` (BooleanField, default=False) - Email verification status
- `role` (CharField, max_length=32, choices, default='entrepreneur')
  - Choices: 'entrepreneur', 'student', 'investor'
- `phone_number` (CharField, max_length=20, blank=True)
- `last_login` (DateTimeField, nullable) - Last login timestamp
- `date_joined` (DateTimeField) - Account creation date (Django default)
- `created_at` (DateTimeField, auto_now_add=True)
- `updated_at` (DateTimeField, auto_now=True)

**Indexes:**
- `email` (for login lookups)
- `username` (for username lookups)
- `role` (for filtering by user type)

**Relationships:**
- One-to-One with `user_profiles` (via `profile` related_name)
- One-to-Many with `startups` (as owner)
- One-to-Many with `applications` (as applicant)
- One-to-Many with `notifications` (as recipient)
- One-to-Many with `favorites` (as user)
- One-to-Many with `interests` (as user)
- One-to-Many with `file_uploads` (as uploader)
- One-to-Many with `messages` (as sender)
- Many-to-Many with `conversations` (as participant)

---

### 2. `user_profiles` (UserProfile)
**Purpose:** Extended optional profile information for users. Separated from core User table for performance and optional data.

**Columns:**
- `user_id` (UUID, Primary Key, Foreign Key → `users.id`) - OneToOne relationship
- `bio` (TextField, blank=True) - User biography
- `location` (CharField, max_length=100, blank=True) - User location
- `website` (URLField, blank=True) - Personal/professional website
- `profile_picture` (ImageField, upload_to='profiles/', blank=True, nullable=True)
- `is_public` (BooleanField, default=False) - Profile visibility setting
- `selected_regions` (JSONField, default=list, blank=True) - Preferred regions (array)
- `skills` (JSONField, default=list, blank=True) - User skills (array)
- `experience` (JSONField, default=list, blank=True) - Work experience (array)
- `references` (JSONField, default=list, blank=True) - Professional references (array)
- `created_at` (DateTimeField, auto_now_add=True)
- `updated_at` (DateTimeField, auto_now=True)

**Relationships:**
- One-to-One with `users` (via `user` ForeignKey)

**Key Difference from User Table:**
- **User table**: Core authentication data (email, password, role, phone) - REQUIRED for login
- **UserProfile table**: Extended optional profile info (bio, skills, experience) - OPTIONAL, created separately
- One user can have at most one profile (OneToOne relationship)
- Profile is not required for account creation

---

## Startup & Job System

### 3. `startups` (Startup)
**Purpose:** Startup listings for both marketplace (for sale) and collaboration (job opportunities) types.

**Columns:**
- `id` (UUID, Primary Key) - Auto-generated UUID
- `owner_id` (UUID, Foreign Key → `users.id`, REQUIRED) - Startup owner/creator
- `title` (CharField, max_length=200, REQUIRED, min_length=5) - Startup name
- `role_title` (CharField, max_length=100, blank=True) - Role title for collaborations
- `description` (TextField, REQUIRED, min_length=20) - Detailed description
- `field` (CharField, max_length=100) - Industry/field
- `website_url` (URLField, blank=True) - Startup website
- `stages` (JSONField, default=list, blank=True) - Development stages (array)
- `type` (CharField, max_length=20, choices, default='marketplace')
  - Choices: 'marketplace' (for sale), 'collaboration' (job opportunities)
- `category` (CharField, max_length=50, choices, default='other')
  - Choices: 'saas', 'ecommerce', 'agency', 'legal', 'marketplace', 'media', 'platform', 'real_estate', 'robotics', 'software', 'web3', 'crypto', 'other'
- `status` (CharField, max_length=20, choices, default='active')
  - Choices: 'active', 'inactive', 'sold', 'paused'
- `views` (IntegerField, default=0) - View counter
- `featured` (BooleanField, default=False) - Featured listing flag

**Financial Fields (for marketplace type):**
- `revenue` (CharField, max_length=50, blank=True)
- `profit` (CharField, max_length=50, blank=True)
- `asking_price` (CharField, max_length=50, blank=True)
- `ttm_revenue` (CharField, max_length=50, blank=True) - Trailing twelve months revenue
- `ttm_profit` (CharField, max_length=50, blank=True) - Trailing twelve months profit
- `last_month_revenue` (CharField, max_length=50, blank=True)
- `last_month_profit` (CharField, max_length=50, blank=True)

**Collaboration Fields (for collaboration type):**
- `earn_through` (CharField, max_length=50, blank=True) - How to earn
- `phase` (CharField, max_length=50, blank=True) - Development phase
- `team_size` (CharField, max_length=50, blank=True) - Current team size

- `created_at` (DateTimeField, auto_now_add=True)
- `updated_at` (DateTimeField, auto_now=True)

**Indexes:**
- `owner` (for filtering by owner)
- `type` (for filtering marketplace vs collaboration)
- `category` (for category filtering)
- `status` (for status filtering)
- `created_at` (for sorting by date)

**Relationships:**
- Many-to-One with `users` (via `owner` ForeignKey)
- One-to-Many with `startup_tags` (via `tags` related_name)
- One-to-Many with `positions` (via `positions` related_name)
- One-to-Many with `applications` (via `applications` related_name)
- One-to-Many with `favorites` (via `favorited_by` related_name)
- One-to-Many with `interests` (via `interests` related_name)

---

### 4. `startup_tags` (StartupTag)
**Purpose:** Tags/labels for categorizing and searching startups.

**Columns:**
- `id` (UUID, Primary Key) - Auto-generated UUID
- `startup_id` (UUID, Foreign Key → `startups.id`, REQUIRED)
- `tag` (CharField, max_length=100, REQUIRED) - Tag name

**Constraints:**
- UNIQUE TOGETHER: (`startup_id`, `tag`) - Prevents duplicate tags per startup

**Indexes:**
- Composite index on (`startup_id`, `tag`)

**Relationships:**
- Many-to-One with `startups` (via `startup` ForeignKey)

---

### 5. `positions` (Position)
**Purpose:** Available job positions/roles within startups (for collaboration type startups).

**Columns:**
- `id` (UUID, Primary Key) - Auto-generated UUID
- `startup_id` (UUID, Foreign Key → `startups.id`, REQUIRED)
- `title` (CharField, max_length=100, REQUIRED) - Position title (e.g., "Frontend Developer")
- `description` (TextField, blank=True) - Position description
- `requirements` (TextField, blank=True) - Job requirements
- `is_active` (BooleanField, default=True) - Whether position is still open
- `created_at` (DateTimeField, auto_now_add=True)

**Indexes:**
- `startup_id` (for filtering positions by startup)

**Relationships:**
- Many-to-One with `startups` (via `startup` ForeignKey)
- One-to-Many with `applications` (via `applications` related_name)

---

### 6. `applications` (Application)
**Purpose:** Job applications submitted by users for positions at startups.

**Columns:**
- `id` (UUID, Primary Key) - Auto-generated UUID
- `startup_id` (UUID, Foreign Key → `startups.id`, REQUIRED)
- `position_id` (UUID, Foreign Key → `positions.id`, REQUIRED)
- `applicant_id` (UUID, Foreign Key → `users.id`, REQUIRED) - User who applied
- `cover_letter` (TextField, blank=True) - Application cover letter
- `experience` (TextField, blank=True) - Applicant's experience
- `portfolio_url` (URLField, blank=True) - Portfolio website URL
- `resume_url` (URLField, blank=True) - Resume/CV file URL
- `status` (CharField, max_length=20, choices, default='pending')
  - Choices: 'pending', 'approved', 'rejected', 'withdrawn'
- `notes` (TextField, blank=True) - Admin/owner internal notes
- `created_at` (DateTimeField, auto_now_add=True)
- `updated_at` (DateTimeField, auto_now=True)

**Constraints:**
- UNIQUE TOGETHER: (`startup_id`, `applicant_id`) - **A user can only apply once per startup** (regardless of position)

**Indexes:**
- `startup_id` (for filtering by startup)
- `applicant_id` (for filtering by applicant)
- `status` (for filtering by status)

**Relationships:**
- Many-to-One with `startups` (via `startup` ForeignKey)
- Many-to-One with `positions` (via `position` ForeignKey)
- Many-to-One with `users` (via `applicant` ForeignKey)

**How Application Links User to Startup:**
- **Direct Links:**
  - `applicant_id` → `users.id` (who applied)
  - `startup_id` → `startups.id` (which startup)
  - `position_id` → `positions.id` (which position)
- **Indirect Link:**
  - Since `positions.startup_id` → `startups.id`, Application connects User to Startup through Position
- **Flow:** User (applicant) → applies to → Position → at → Startup
- **Constraint:** One user can only have one application per startup (even if multiple positions exist)

---

## User Engagement

### 7. `favorites` (Favorite)
**Purpose:** Users (typically investors) saving/bookmarking startups they're interested in.

**Columns:**
- `id` (UUID, Primary Key) - Auto-generated UUID
- `user_id` (UUID, Foreign Key → `users.id`, REQUIRED)
- `startup_id` (UUID, Foreign Key → `startups.id`, REQUIRED)
- `created_at` (DateTimeField, auto_now_add=True) - When favorited

**Constraints:**
- UNIQUE TOGETHER: (`user_id`, `startup_id`) - **A user can only favorite a startup once**

**Indexes:**
- `user_id` (for finding user's favorites)
- `startup_id` (for finding who favorited a startup)

**Relationships:**
- Many-to-One with `users` (via `user` ForeignKey)
- Many-to-One with `startups` (via `startup` ForeignKey)

**Relationship Type:**
- **Many-to-Many relationship** between User and Startup (implemented via Favorite junction table)
- One user can favorite many startups
- One startup can be favorited by many users
- Used primarily for investor engagement (saving startups for later review)

---

### 8. `interests` (Interest)
**Purpose:** Investors expressing formal interest in a startup (more serious than favorite).

**Columns:**
- `id` (UUID, Primary Key) - Auto-generated UUID
- `user_id` (UUID, Foreign Key → `users.id`, REQUIRED) - Investor
- `startup_id` (UUID, Foreign Key → `startups.id`, REQUIRED)
- `message` (TextField, blank=True) - Optional interest message
- `created_at` (DateTimeField, auto_now_add=True)

**Constraints:**
- UNIQUE TOGETHER: (`user_id`, `startup_id`) - **A user can only express interest once per startup**

**Indexes:**
- `user_id` (for finding user's interests)
- `startup_id` (for finding who's interested in a startup)

**Relationships:**
- Many-to-One with `users` (via `user` ForeignKey)
- Many-to-One with `startups` (via `startup` ForeignKey)

---

### 9. `notifications` (Notification)
**Purpose:** In-app notifications sent to users for various events.

**Columns:**
- `id` (UUID, Primary Key) - Auto-generated UUID
- `user_id` (UUID, Foreign Key → `users.id`, REQUIRED) - Notification recipient
- `type` (CharField, max_length=50, choices, REQUIRED)
  - Choices: 'application_status', 'new_application', 'pitch', 'general'
- `title` (CharField, max_length=200, REQUIRED) - Notification title
- `message` (TextField, blank=True) - Notification message body
- `data` (JSONField, default=dict, blank=True) - Additional structured data
- `is_read` (BooleanField, default=False) - Read status
- `created_at` (DateTimeField, auto_now_add=True)

**Indexes:**
- Composite index on (`user_id`, `is_read`) - For finding unread notifications
- `created_at` (for sorting by date)

**Relationships:**
- Many-to-One with `users` (via `user` ForeignKey)

---

## Messaging System

### 10. `conversations` (Conversation)
**Purpose:** Chat conversations between multiple users.

**Columns:**
- `id` (UUID, Primary Key) - Auto-generated UUID
- `title` (CharField, max_length=200, blank=True) - Optional conversation title
- `is_active` (BooleanField, default=True) - Whether conversation is active
- `created_at` (DateTimeField, auto_now_add=True)
- `updated_at` (DateTimeField, auto_now=True)

**Indexes:**
- `created_at` (for sorting conversations)
- `is_active` (for filtering active conversations)

**Relationships:**
- Many-to-Many with `users` (via `participants` ManyToManyField, junction table: `conversations_participants`)
- One-to-Many with `messages` (via `messages` related_name)

**Junction Table: `conversations_participants`**
- `conversation_id` (UUID, Foreign Key → `conversations.id`)
- `user_id` (UUID, Foreign Key → `users.id`)
- Composite Primary Key on both fields

---

### 11. `messages` (Message)
**Purpose:** Individual messages within conversations.

**Columns:**
- `id` (UUID, Primary Key) - Auto-generated UUID
- `conversation_id` (UUID, Foreign Key → `conversations.id`, REQUIRED)
- `sender_id` (UUID, Foreign Key → `users.id`, REQUIRED) - Message sender
- `content` (TextField, REQUIRED) - Message text content
- `message_type` (CharField, max_length=20, choices, default='text')
  - Choices: 'text', 'image', 'file'
- `attachment` (FileField, upload_to='messages/', blank=True, nullable=True) - File attachment
- `is_read` (BooleanField, default=False) - Read status
- `created_at` (DateTimeField, auto_now_add=True)

**Indexes:**
- Composite index on (`conversation_id`, `created_at`) - For ordering messages in conversation
- `sender_id` (for filtering by sender)
- `is_read` (for finding unread messages)

**Default Ordering:**
- Ordered by `created_at` (ascending) - Oldest first

**Relationships:**
- Many-to-One with `conversations` (via `conversation` ForeignKey)
- Many-to-One with `users` (via `sender` ForeignKey)

---

### 12. `file_uploads` (FileUpload)
**Purpose:** File uploads for various purposes (resumes, profile pictures, startup images, message attachments).

**Columns:**
- `id` (UUID, Primary Key) - Auto-generated UUID
- `user_id` (UUID, Foreign Key → `users.id`, REQUIRED) - User who uploaded
- `file` (FileField, upload_to='uploads/', REQUIRED) - File path
- `file_type` (CharField, max_length=20, choices, REQUIRED)
  - Choices: 'resume', 'startup_image', 'profile_picture', 'message_attachment', 'other'
- `original_name` (CharField, max_length=255, REQUIRED) - Original filename
- `file_size` (BigIntegerField, REQUIRED) - File size in bytes
- `mime_type` (CharField, max_length=100, REQUIRED) - MIME type
- `is_active` (BooleanField, default=True) - Whether file is active/available
- `created_at` (DateTimeField, auto_now_add=True)

**Indexes:**
- `user_id` (for finding user's uploads)
- `file_type` (for filtering by file type)
- `created_at` (for sorting by date)

**Relationships:**
- Many-to-One with `users` (via `user` ForeignKey)

---

## Key Relationships Explained

### User vs UserProfile
- **User**: Core authentication table with email, password, role, phone_number. Required for login.
- **UserProfile**: Extended optional profile with bio, skills, experience, location. Not required for account creation.
- **Relationship**: OneToOne - Each user can have at most one profile (optional).
- **Why separate**: Performance (profile data not needed for every query) and optional nature of profile data.

### Favorite Table
- **Purpose**: Many-to-Many relationship between User and Startup (junction table).
- **Columns**: `id`, `user_id` (FK), `startup_id` (FK), `created_at`.
- **Constraint**: UNIQUE (`user_id`, `startup_id`) - User can only favorite a startup once.
- **Use Case**: Investors saving/bookmarking startups for later review.

### Application Relations
- **Links**: User (applicant) → Position → Startup
- **Foreign Keys**:
  - `applicant_id` → `users.id` (who applied)
  - `position_id` → `positions.id` (which position)
  - `startup_id` → `startups.id` (which startup)
- **Constraint**: UNIQUE (`startup_id`, `applicant_id`) - User can only apply once per startup (regardless of position).
- **Flow**: User applies to a Position at a Startup. Since Position belongs to Startup, Application indirectly links User to Startup through Position.

---

## Entity Relationship Summary

```
User (1) ──< (Many) Startup (owned by)
User (1) ──< (Many) Application (applied by)
User (1) ──< (Many) Favorite (favorited by)
User (1) ──< (Many) Interest (expressed by)
User (1) ──< (Many) Notification (received by)
User (1) ──< (Many) FileUpload (uploaded by)
User (1) ──< (Many) Message (sent by)
User (1) ──|| (One) UserProfile (has profile)
User (Many) ──< (Many) Conversation (participates in)

Startup (1) ──< (Many) StartupTag (has tags)
Startup (1) ──< (Many) Position (has positions)
Startup (1) ──< (Many) Application (receives applications)
Startup (1) ──< (Many) Favorite (favorited in)
Startup (1) ──< (Many) Interest (interested in)

Position (1) ──< (Many) Application (for position)

Conversation (1) ──< (Many) Message (contains messages)
```

---

## Database Technology
- **ORM**: Django ORM
- **Database**: SQLite (db.sqlite3) - Can be migrated to PostgreSQL/MySQL
- **UUID**: All primary keys use UUID v4 (except UserProfile which uses BigAutoField for user_id)

---

## Notes
- All timestamps use `auto_now_add=True` for creation and `auto_now=True` for updates
- All UUID fields are auto-generated and non-editable
- Foreign keys use `CASCADE` delete (deleting parent deletes children)
- Many-to-Many relationships use junction tables (Conversation-User, User-Groups, User-Permissions)
- JSON fields store arrays/objects for flexible data (stages, skills, experience, etc.)

