# Find Users for Two-Tower Model - Quick Guide

## Overview

This guide shows you how to find users in the database who have more than 5 interactions and optionally generate two-tower recommendations for them.

## Command: `find_users_for_two_tower`

### Basic Usage

Find all users with more than 5 interactions:
```bash
cd backend
python manage.py find_users_for_two_tower
```

### Options

- `--min-interactions N`: Set minimum number of interactions (default: 5)
  ```bash
  python manage.py find_users_for_two_tower --min-interactions 10
  ```

- `--generate-recommendations`: Generate two-tower recommendations for found users
  ```bash
  python manage.py find_users_for_two_tower --generate-recommendations
  ```

- `--limit N`: Number of recommendations per user (default: 10)
  ```bash
  python manage.py find_users_for_two_tower --generate-recommendations --limit 20
  ```

- `--output PATH`: Save results to a JSON file
  ```bash
  python manage.py find_users_for_two_tower --output results.json
  ```

### Examples

1. **Find users with more than 5 interactions:**
   ```bash
   python manage.py find_users_for_two_tower --min-interactions 5
   ```

2. **Find users and generate recommendations:**
   ```bash
   python manage.py find_users_for_two_tower --min-interactions 5 --generate-recommendations --limit 10
   ```

3. **Save results to file:**
   ```bash
   python manage.py find_users_for_two_tower --min-interactions 5 --output users_with_interactions.json
   ```

## Output Format

The command displays:
- User list with username, email, role, and interaction count
- Interaction type breakdown (view, click, like, apply, favorite, interest, dislike)
- Summary by role (entrepreneur, student, investor)
- If `--generate-recommendations` is used: top recommendations for each user

## Current Statistics

Based on the latest run:
- **Total users with >5 interactions:** 349
- **By role:**
  - Entrepreneurs: 141 users
  - Investors: 53 users
  - Students: 155 users

## Two-Tower Model Requirements

To generate recommendations, you need:
1. A trained two-tower model in `recommendation_service/models/two_tower_v1.pth`
2. The `inference_two_tower.py` module accessible

If the model is not found, the command will still list users but skip recommendation generation.

## Notes

- Users are sorted by interaction count (highest first)
- The command shows interaction type breakdown for each user
- Recommendation generation is limited to the first 20 users for display purposes
- All users are processed if saving to a file

