# Synthetic Data Generation Prompt for Startup Recommendation System

You are generating realistic synthetic data for a startup recommendation system built with Django.

## Goal

Create SQL seed data (or Django fixtures/JSONs) for these tables:

1. **users** (api.User)
2. **user_profiles** (api.UserProfile)
3. **user_onboarding_preferences** (api.UserOnboardingPreferences)
4. **startups** (api.Startup)
5. **startup_tags** (api.StartupTag)
6. **positions** (api.Position)
7. **user_interactions** (api.UserInteraction)

The goal is to have high-quality, meaningful data that can be used to train and test a recommendation model (ALS or Two-Tower).

---

## 1️⃣ USERS TABLE (api.User)

**Columns:**
- `id` (UUID, primary key) - Auto-generated UUID
- `username` (string, required)
- `email` (string, unique, required) - Used as USERNAME_FIELD
- `password` (string, hashed) - Use Django's password hashing
- `role` (CharField, choices, required) - One of:
  - `'entrepreneur'` - Entrepreneurs/Founders
  - `'student'` - Students/Professionals
  - `'investor'` - Investors
- `phone_number` (string, optional)
- `is_active` (boolean, default=True)
- `email_verified` (boolean, default=False)
- `created_at` (DateTime, auto_now_add)
- `updated_at` (DateTime, auto_now)

**Generate around 300-400 users**, distributed across roles:
- ~40% entrepreneurs
- ~45% students
- ~15% investors

---

## 2️⃣ USER_PROFILES TABLE (api.UserProfile)

**Columns:**
- `user_id` (UUID, OneToOne → users.id, required)
- `bio` (TextField, optional) - Short professional bio
- `location` (string, max_length=100, optional) - City/Country
- `website` (URL, optional)
- `selected_regions` (JSONField, list) - Geographic preferences
- `skills` (JSONField, list) - Technical/business skills, e.g., ['Python', 'React', 'Marketing', 'Product Management']
- `experience` (JSONField, list) - Work experience entries
- `preferred_work_modes` (JSONField, list) - e.g., ['remote', 'hybrid', 'onsite']
- `preferred_compensation_types` (JSONField, list) - e.g., ['paid', 'equity', 'both']
- `onboarding_completed` (boolean, default=False)
- `created_at` (DateTime)
- `updated_at` (DateTime)

**Generate profiles for all users** with:
- Skills relevant to their role (developers: technical skills, entrepreneurs: business skills, investors: finance/strategy)
- Realistic work mode preferences (students prefer remote/part-time, entrepreneurs prefer equity)
- Location diversity

---

## 3️⃣ USER_ONBOARDING_PREFERENCES TABLE (api.UserOnboardingPreferences)

**Columns:**
- `id` (UUID, primary key)
- `user_id` (UUID, OneToOne → users.id, required)
- `selected_categories` (JSONField, list) - From Startup.CATEGORY_CHOICES:
  - `'saas'`, `'ecommerce'`, `'agency'`, `'legal'`, `'marketplace'`, `'media'`, `'platform'`, `'real_estate'`, `'robotics'`, `'software'`, `'web3'`, `'crypto'`, `'other'`
- `selected_fields` (JSONField, list) - Industry/field strings, e.g., ['fintech', 'healthcare', 'education', 'ecommerce']
- `selected_tags` (JSONField, list) - Tag strings that match startup_tags.tag values
- `preferred_startup_stages` (JSONField, list) - Stage strings, e.g., ['early', 'growth', 'mature', 'scaling']
- `preferred_engagement_types` (JSONField, list) - e.g., ['full-time', 'part-time', 'equity', 'paid']
- `preferred_skills` (JSONField, list) - Skills they want to work with, e.g., ['Python', 'React', 'Machine Learning']
- `onboarding_completed` (boolean, default=True for seed data)
- `created_at` (DateTime)
- `updated_at` (DateTime)

**Generate preferences for all users** that align with their role and profile:
- Entrepreneurs: SaaS, marketplace, platform categories; equity engagement; business skills
- Students: Software, web3, media categories; part-time/paid engagement; technical skills
- Investors: Fintech, healthcare, platform categories; growth/mature stages; no skills needed

---

## 4️⃣ STARTUPS TABLE (api.Startup)

**Columns:**
- `id` (UUID, primary key)
- `owner_id` (UUID, ForeignKey → users.id, required) - Must be an entrepreneur user
- `title` (string, max_length=200, required, min_length=5) - Startup name
- `role_title` (string, optional) - For collaboration type startups
- `description` (TextField, required, min_length=20) - Detailed description
- `field` (string, max_length=100, required) - Industry/field, e.g., 'fintech', 'healthcare', 'education'
- `website_url` (URL, optional)
- `stages` (JSONField, list) - Development stages, e.g., ['early', 'growth', 'mature', 'scaling']
- `type` (CharField, choices, default='marketplace') - One of:
  - `'marketplace'` - Startup for sale
  - `'collaboration'` - Job opportunities/collaboration
- `category` (CharField, choices, required) - One of:
  - `'saas'`, `'ecommerce'`, `'agency'`, `'legal'`, `'marketplace'`, `'media'`, `'platform'`, `'real_estate'`, `'robotics'`, `'software'`, `'web3'`, `'crypto'`, `'other'`
- `status` (CharField, choices, default='active') - One of:
  - `'active'`, `'inactive'`, `'sold'`, `'paused'`
- `views` (integer, default=0) - View counter
- `featured` (boolean, default=False)
- `revenue` (string, optional) - For marketplace type
- `profit` (string, optional) - For marketplace type
- `asking_price` (string, optional) - For marketplace type
- `earn_through` (string, optional) - For collaboration type
- `phase` (string, optional) - For collaboration type
- `team_size` (string, optional) - For collaboration type
- `created_at` (DateTime)
- `updated_at` (DateTime)

**Generate 200-300 startups** with:
- All owned by entrepreneur users
- Mix of 'marketplace' and 'collaboration' types (~60% collaboration, 40% marketplace)
- Diversity across all categories
- Realistic field values matching selected_fields from user preferences
- Stages as JSON arrays (most have 1-2 stages)
- ~90% with status='active'
- Some featured startups (5-10%)

---

## 5️⃣ STARTUP_TAGS TABLE (api.StartupTag)

**Columns:**
- `id` (UUID, primary key)
- `startup_id` (UUID, ForeignKey → startups.id, required)
- `tag` (string, max_length=100, required)
- Unique constraint: (`startup_id`, `tag`)

**Common tag examples:**
- Technology: `'AI'`, `'Machine Learning'`, `'blockchain'`, `'mobile'`, `'web'`, `'cloud'`, `'IoT'`, `'cybersecurity'`
- Business model: `'B2B'`, `'B2C'`, `'SaaS'`, `'marketplace'`, `'platform'`
- Stage/Status: `'remote-first'`, `'bootstrapped'`, `'funded'`, `'YC-backed'`, `'scalable'`
- Industry: `'fintech'`, `'healthtech'`, `'edtech'`, `'ecommerce'`, `'automation'`

**Generate 3-8 tags per startup** (average 5):
- Each startup should have tags relevant to its category, field, and description
- Ensure tag diversity across startups
- Some tags should appear frequently (popular tags), others rarely (niche tags)

---

## 6️⃣ POSITIONS TABLE (api.Position)

**Columns:**
- `id` (UUID, primary key)
- `startup_id` (UUID, ForeignKey → startups.id, required)
- `title` (string, max_length=100, required) - Job title, e.g., 'Backend Developer', 'Product Manager', 'Marketing Lead'
- `description` (TextField, optional) - Position description
- `requirements` (TextField, optional) - Required skills/experience
- `is_active` (boolean, default=True)
- `created_at` (DateTime)

**Generate positions only for 'collaboration' type startups:**
- 1-5 positions per collaboration startup (average 2-3)
- Position titles should match skills in user profiles (developers, designers, marketers, etc.)
- Requirements should list relevant technical/business skills
- ~80% with is_active=True

---

## 7️⃣ USER_INTERACTIONS TABLE (api.UserInteraction)

**Columns:**
- `id` (UUID, primary key)
- `user_id` (UUID, ForeignKey → users.id, required)
- `startup_id` (UUID, ForeignKey → startups.id, required)
- `interaction_type` (CharField, choices, required) - One of:
  - `'view'` (weight: 0.5)
  - `'click'` (weight: 1.0)
  - `'like'` (weight: 2.0)
  - `'dislike'` (weight: -1.0)
  - `'favorite'` (weight: 2.5)
  - `'apply'` (weight: 3.0) - Only for collaboration startups with positions
  - `'interest'` (weight: 3.5) - Typically for investors
- `position_id` (UUID, ForeignKey → positions.id, optional, nullable) - Only for 'apply' interactions
- `weight` (FloatField) - Auto-calculated based on interaction_type
- `metadata` (JSONField, optional) - Additional context
- `created_at` (DateTime) - Distributed across last 6 months
- Unique constraint: (`user_id`, `startup_id`, `interaction_type`)

**Generate 5,000-10,000 interactions** such that:

### Preference Alignment (70-80% of interactions):
- Users interact more with startups matching their preferences:
  - **Students**: Prefer startups with categories in their `selected_categories`, skills matching their `preferred_skills`, part-time/paid engagement
  - **Entrepreneurs**: Prefer startups in similar categories/fields, equity engagement, early/growth stages
  - **Investors**: Prefer growth/mature stage startups, funded/bootstrapped tags, high-revenue marketplace startups

### Interaction Type Distribution:
- `view`: 40-50% of interactions (most common)
- `click`: 20-25% of interactions
- `like`: 10-15% of interactions
- `favorite`: 5-10% of interactions (more common for investors)
- `apply`: 5-10% of interactions (only students/entrepreneurs, only collaboration startups with positions)
- `interest`: 3-5% of interactions (mostly investors)
- `dislike`: 2-5% of interactions (negative signal)

### Realistic Patterns:
- Users often have multiple interactions with the same startup (e.g., view → click → like → favorite)
- Some users are very active (10-50 interactions), others are passive (1-5 interactions)
- Popular startups (high views) get more interactions overall
- Recent startups (created in last 3 months) get more recent interactions
- Add 10-20% noise (interactions that don't perfectly match preferences)

### Timestamp Distribution:
- Spread interactions across last 6 months
- More recent interactions (last 1-2 months) should be more frequent
- Use realistic time patterns (more activity on weekdays, less on weekends)

---

## 8️⃣ RELATION LOGIC & CORRELATIONS

### Role-Based Preferences:

**Entrepreneurs:**
- Prefer: SaaS, marketplace, platform categories
- Stages: early, growth, scaling
- Engagement: equity, full-time
- Skills: Business Development, Product Management, Marketing, Sales
- More likely to: view, click, like startups in similar domains

**Students:**
- Prefer: software, web3, media, saas categories
- Stages: early, growth
- Engagement: part-time, paid, equity
- Skills: Python, React, JavaScript, Machine Learning, Node.js, Django
- More likely to: view, click, apply to collaboration startups with matching skills

**Investors:**
- Prefer: fintech, healthcare, platform, saas categories
- Stages: growth, mature, scaling
- Engagement: equity (only)
- Skills: None (or finance, strategy)
- More likely to: view, favorite, express interest in high-potential startups
- Prefer funded/bootstrapped startups with revenue

### Startup-User Matching:
- Collaboration startups with positions → Students/Entrepreneurs apply
- Marketplace startups with revenue → Investors favorite/express interest
- Startups with tags matching user preferred_skills → Higher interaction rates
- Startups in user's selected_categories → More views/clicks

---

## 9️⃣ VALIDATION REQUIREMENTS

- **Foreign Key Integrity**: All foreign keys must reference existing records
  - `user_interactions.user_id` → existing users
  - `user_interactions.startup_id` → existing startups
  - `user_interactions.position_id` → existing positions (if not null)
  - `startups.owner_id` → entrepreneur users only
  - `startup_tags.startup_id` → existing startups
  - `positions.startup_id` → existing startups (preferably collaboration type)
  - `user_profiles.user_id` → existing users (OneToOne, all users should have profile)
  - `user_onboarding_preferences.user_id` → existing users (OneToOne, all users should have preferences)

- **Data Consistency**:
  - All users have a user_profile
  - All users have user_onboarding_preferences
  - All startups have at least 3 tags
  - Collaboration startups have at least 1 position
  - Apply interactions only reference collaboration startups with positions
  - Interest interactions mostly from investors
  - Unique constraints respected (no duplicate user-startup-interaction_type combinations)

- **Timestamp Distribution**:
  - User created_at: Spread across last 12 months
  - Startup created_at: Spread across last 18 months
  - Interaction created_at: Spread across last 6 months, weighted toward recent dates

- **Weight Values**:
  - Must match interaction_type: view=0.5, click=1.0, like=2.0, dislike=-1.0, favorite=2.5, apply=3.0, interest=3.5

---

## 🔟 GOAL OF DATA

The resulting dataset should:

✅ Be meaningful enough to train a matrix-factorization (ALS) or two-tower model  
✅ Produce realistic recommendations where user preferences correlate with startup attributes  
✅ Be diverse enough for testing cold-start and warm-start users  
✅ Have sufficient interaction density (average 15-30 interactions per user)  
✅ Include realistic preference alignment (70-80% match rate)  
✅ Support testing of different recommendation scenarios:
  - Developer → Startup recommendations (students applying to collaboration startups)
  - Investor → Startup recommendations (investors favoriting marketplace startups)
  - Entrepreneur → Startup recommendations (entrepreneurs viewing similar startups)

---

## 📝 OUTPUT FORMAT

Generate data in one of these formats:
1. **Django Fixtures (JSON)** - Preferred for Django projects
2. **SQL INSERT statements** - For direct database import
3. **CSV files** - One per table, with proper UUID handling
4. **Python script** - Using Django ORM to create objects programmatically

Include:
- All required fields populated
- Realistic text content (descriptions, bios, etc.)
- Proper UUID generation
- Valid email addresses
- Hashed passwords (if using Django fixtures, use `pbkdf2_sha256` format)
- Realistic timestamps with proper distribution

---

## 🎯 DATA QUALITY CHECKLIST

Before finalizing, ensure:
- [ ] All foreign keys are valid
- [ ] No duplicate unique constraint violations
- [ ] Interaction weights match interaction types
- [ ] Preference alignment is visible (users interact with matching startups)
- [ ] Sufficient data diversity (not all users have identical preferences)
- [ ] Realistic interaction patterns (some power users, some passive users)
- [ ] Timestamps are chronologically consistent (users created before their interactions)
- [ ] All required fields are populated
- [ ] Text fields have realistic content (not just "test" or "lorem ipsum")
- [ ] Email addresses are unique and valid format
- [ ] UUIDs are properly formatted

