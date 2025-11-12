# Complete Use Case Testing Guide 🧪0

## Prerequisites
- ✅ Backend server running on `http://127.0.0.1:8000`
- ✅ Frontend server running on `http://localhost:5174`
- ✅ Database migrations applied
- ✅ Both servers accessible

---

## 🧪 UC1: User Registration and Authentication

### Test Scenario 1: New User Registration (Entrepreneur)
1. **Navigate to Registration**
   - Open browser: `http://localhost:5174`
   - Click "Signup" or navigate to `/signup`

2. **Fill Registration Form**
   ```
   Username: john_entrepreneur
   Email: john@example.com
   Password: password123
   Role: Entrepreneur
   ```

3. **Submit and Verify**
   - Click "Signup"
   - Should see success toast: "Signup Successfully"
   - Should redirect to email verification page
   - Check browser console for verification code (or check backend logs)

4. **Email Verification**
   - Enter the 6-digit verification code
   - Click "Verify Email"
   - Should see "Email verified successfully!"
   - Should redirect to login page

### Test Scenario 2: User Login
1. **Login with verified account**
   - Email: `john@example.com`
   - Password: `password123`
   - Click "Login"

2. **Verify Success**
   - Should see "Login Successfully" toast
   - Should redirect to `/dashboard`
   - Should see entrepreneur-specific dashboard
   - Navbar should show user role: "john_entrepreneur (entrepreneur)"

### Test Scenario 3: Role-Based Access
1. **Test Entrepreneur Access**
   - Navigate to `/createstartup` - Should work ✅
   - Navigate to `/pitch-idea` - Should work ✅
   - Navigate to `/investor-dashboard` - Should show "Access Denied" ❌

2. **Test Without Authentication**
   - Logout and try to access `/dashboard` - Should redirect to `/login`

---

## 🧪 UC2: Create Startup Listings

### Test Scenario 1: Create Marketplace Startup
1. **Access Creation Form**
   - Login as entrepreneur
   - Navigate to `/createstartup` or click "Create Startup" in navbar

2. **Fill Required Fields**
   ```
   Startup Name: TechFlow Solutions
   Role Title: Co-founder & CTO
   Category: SaaS (select radio button)
   Description: AI-powered workflow automation platform for businesses
   Field/Industry: Technology
   ```

3. **Select Stages**
   - Check: ✅ MVP Stage
   - Check: ✅ Product Market Fit
   - Check: ✅ Fund raising

4. **Set Startup Type**
   - Select: "Marketplace (For Sale)"

5. **Fill Marketplace Information**
   ```
   Website URL: https://techflow.com
   Current Phase: Series A
   Team Size: 5-8 people
   How do you earn?: SaaS Subscriptions
   ```

6. **Optional Financial Data**
   ```
   Current Revenue: $50,000/month
   Current Profit: $20,000/month
   Asking Price: $2,500,000
   TTM Revenue: $600,000
   TTM Profit: $240,000
   ```

7. **Submit and Verify**
   - Click "Create Project"
   - Should see "Startup created successfully!" toast
   - Should redirect to `/dashboard`

### Test Scenario 2: Create Collaboration Startup
1. **Repeat process but select "Collaboration (Looking for Team)"**
2. **Fill collaboration-specific fields**
   ```
   Current Team Size: Just me
   How will team members earn?: Equity + Revenue Share
   ```

---

## 🧪 UC3: Search Startups

### Test Scenario 1: Basic Search
1. **Access Search Page**
   - Navigate to `/search` or click search in navbar

2. **Test Text Search**
   - Enter search term: "TechFlow"
   - Click "Search"
   - Should show the startup you created

3. **Test Filters**
   - Select Type: "For Sale"
   - Select Category: "SaaS"
   - Select Phase: "Series A"
   - Should filter results accordingly

4. **Test Clear Filters**
   - Click "Clear Filters"
   - Should reset all filters and show all results

### Test Scenario 2: Marketplace Browsing
1. **Navigate to `/marketplace`**
2. **Test marketplace-specific filters**
   - Sort by: Revenue
   - Filter by type, min/max revenue
3. **Verify startup cards display properly**

---

## 🧪 UC4: Join a Startup

### Test Scenario 1: Register as Student
1. **Create new account**
   ```
   Username: jane_student
   Email: jane@example.com
   Password: password123
   Role: Student/Professional
   ```

2. **Complete verification and login**

### Test Scenario 2: Apply for Position
1. **Find startup with positions**
   - Navigate to `/collaboration`
   - Click on a startup card
   - Should see startup details

2. **Apply for position**
   - Click "Apply for this Position" (if available)
   - Fill application form:
   ```
   Cover Letter: I'm interested in joining your team because...
   Experience & Skills: 3 years React development, Node.js...
   Portfolio URL: https://jane-portfolio.com
   ```
   - Optionally upload resume file

3. **Submit Application**
   - Click "Submit Application"
   - Should see "Application submitted successfully!"
   - Should redirect to dashboard

---

## 🧪 UC5: Recruit Team Members

### Test Scenario 1: Create Position (As Entrepreneur)
1. **Access Position Management**
   - Login as entrepreneur
   - Navigate to a startup you own
   - Click "Manage Positions" or go to `/startups/{id}/positions`

2. **Create New Position**
   - Click "+ Create New Position"
   - Fill form:
   ```
   Position Title: Frontend Developer
   Job Description: We're looking for a React developer to help build...
   Requirements: 2+ years React, TypeScript experience preferred
   ```

3. **Submit Position**
   - Click "Create Position"
   - Should see position in the list
   - Should show "0 Applications" initially

### Test Scenario 2: Manage Applications
1. **Wait for applications** (from UC4 testing)
2. **View applications**
   - Click "View Applications" on position
   - Should see list of applicants

3. **Review application**
   - Click on application to view details
   - Should see cover letter, experience, etc.

---

## 🧪 UC6: Investor Engagement

### Test Scenario 1: Register as Investor
1. **Create investor account**
   ```
   Username: mike_investor
   Email: mike@investor.com
   Password: password123
   Role: Investor
   ```

### Test Scenario 2: Express Interest
1. **Browse marketplace**
   - Navigate to `/marketplace`
   - Click on a startup for sale

2. **Engage with startup**
   - Click "🤍 Add to Favorites"
   - Should change to "❤️ Favorited"

3. **Express Interest**
   - Fill interest message: "I'm interested in investing $500K..."
   - Click "Express Interest"
   - Should see "Interest expressed successfully!"

### Test Scenario 3: Investor Dashboard
1. **Navigate to `/investor-dashboard`**
2. **Check tabs**
   - "My Favorites" - should show favorited startups
   - "My Interests" - should show expressed interests
   - "Investment Opportunities" - should show available startups

---

## 🧪 UC7: Pitch Business Ideas

### Test Scenario 1: Create Pitch (As Entrepreneur)
1. **Access Pitch Form**
   - Login as entrepreneur
   - Navigate to `/pitch-idea` or click "Pitch Idea" in navbar

2. **Fill Pitch Information**
   ```
   Business Idea Title: GreenTech Analytics
   Executive Summary: AI-powered sustainability tracking platform...
   Problem Statement: Companies struggle to measure their carbon footprint...
   Solution: Our platform provides real-time environmental impact analysis...
   Market Size: $2.5B sustainability software market
   Business Model: SaaS subscription with tiered pricing
   Funding Needed: $1,000,000
   Use of Funds: Product development 60%, Marketing 30%, Team 10%
   ```

3. **Add Supporting Materials**
   ```
   Pitch Deck URL: https://docs.google.com/presentation/pitch-deck
   Video Pitch URL: https://youtube.com/watch?v=pitch-video
   Contact Email: entrepreneur@greentech.com
   Contact Phone: +1-555-123-4567
   ```

4. **Select Investors**
   - Check boxes for available investors
   - Should see investor list if any investors are registered

5. **Submit Pitch**
   - Click "Submit Pitch to Investors"
   - Should see "Pitch submitted successfully!"

---

## 🧪 UC8: Buy and Sell Startups

### Test Scenario 1: List Startup for Sale
1. **This is covered in UC2** - when creating marketplace startup

### Test Scenario 2: Browse and Purchase Interest
1. **As investor, browse marketplace**
   - Should see startups with financial information
   - TTM Revenue, TTM Profit, Asking Price should display

2. **Express purchase interest**
   - Click startup card
   - View financial metrics
   - Use messaging or interest system to contact seller

---

## 🧪 Messaging System Testing

### Test Scenario 1: Send Messages
1. **Access messaging**
   - Navigate to `/message`
   - Should see "Conversations" and "People" tabs

2. **Start new conversation**
   - Click "People" tab
   - Click on a user
   - Should create/open conversation

3. **Send message**
   - Type message: "Hi, I'm interested in your startup"
   - Press Enter or click Send
   - Should see message appear

4. **Test as different user**
   - Login as different user
   - Check messages
   - Reply to conversation

---

## 🧪 Role-Based UI Testing

### Test Each Role's Navbar
1. **Entrepreneur** should see:
   - Dashboard, Marketplace, Collaboration, Messages, Account
   - Create Startup, Pitch Idea

2. **Student** should see:
   - Dashboard, Marketplace, Collaboration, Messages, Account
   - Find Jobs

3. **Investor** should see:
   - Dashboard, Marketplace, Collaboration, Messages, Account
   - Investor Panel

### Test Role-Based Access
1. **Try accessing restricted pages**
   - Entrepreneur accessing `/investor-dashboard` → Access Denied
   - Student accessing `/createstartup` → Access Denied
   - Investor accessing `/pitch-idea` → Access Denied

---

## 🛠️ Debugging Tips

### Check Browser Console
```javascript
// Open Developer Tools (F12)
// Check for errors in Console tab
// Look for network requests in Network tab
```

### Check Backend Logs
```bash
# In backend terminal, look for API calls
# Should see POST/GET requests with status codes
```

### Test API Directly
```bash
# Test authentication
curl -X POST http://127.0.0.1:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"john@example.com","password":"password123"}'

# Test marketplace
curl http://127.0.0.1:8000/api/marketplace
```

### Common Issues & Solutions
1. **CORS errors**: Check backend CORS settings
2. **Authentication errors**: Check token in localStorage
3. **404 errors**: Check URL patterns match exactly
4. **Form submission fails**: Check required fields
5. **Images not loading**: Check public folder paths

---

## ✅ Testing Checklist

- [ ] UC1: Registration, verification, login for all 3 roles
- [ ] UC2: Create both marketplace and collaboration startups
- [ ] UC3: Search with filters, browse marketplace and collaboration
- [ ] UC4: Apply for positions as student
- [ ] UC5: Create positions and manage applications as entrepreneur
- [ ] UC6: Express interest and manage favorites as investor
- [ ] UC7: Submit business pitch as entrepreneur
- [ ] UC8: Browse marketplace for purchase opportunities
- [ ] Messaging: Send/receive messages between users
- [ ] Role-based access: Test restricted routes
- [ ] UI responsiveness: Test on different screen sizes

**Complete all items above to verify 100% functionality!** 🎉