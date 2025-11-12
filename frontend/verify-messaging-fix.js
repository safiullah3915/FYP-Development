// Simple verification script for messaging components
// Run this in your frontend project root

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

console.log("🔍 Verifying messaging components fixes...");

const checkFile = (filePath, description) => {
  try {
    const fullPath = path.join(__dirname, filePath);
    const content = fs.readFileSync(fullPath, 'utf8');
    
    console.log(`✅ ${description} - File exists and readable`);
    
    // Check for specific imports
    if (content.includes("import { useAuth }")) {
      console.log(`  ✅ useAuth import found`);
    } else {
      console.log(`  ❌ useAuth import not found`);
    }
    
    if (content.includes("import { Navbar }")) {
      console.log(`  ✅ Navbar import found`);
    } else {
      console.log(`  ❌ Navbar import not found`);
    }
    
    if (content.includes("<Navbar />") || content.includes("<Navbar/>")) {
      console.log(`  ✅ Navbar component usage found`);
    } else {
      console.log(`  ❌ Navbar component usage not found`);
    }
    
    if (content.includes("calc(100vh - 70px)")) {
      console.log(`  ✅ CSS adjusted for navbar height`);
    } else {
      console.log(`  ⚠️  CSS might need adjustment for navbar`);
    }
    
    console.log('');
    
  } catch (error) {
    console.log(`❌ ${description} - Error: ${error.message}`);
  }
};

// Check both messaging components
checkFile('src/pages/message/Message.jsx', 'Message.jsx');
checkFile('src/pages/message/MessageDark.jsx', 'MessageDark.jsx');

// Check AuthContext exports
try {
  const authContextPath = path.join(__dirname, 'src/contexts/AuthContext.jsx');
  const authContent = fs.readFileSync(authContextPath, 'utf8');
  
  console.log(`✅ AuthContext.jsx - File exists and readable`);
  
  if (authContent.includes("export { AuthContext }")) {
    console.log(`  ✅ AuthContext export found`);
  } else {
    console.log(`  ❌ AuthContext export not found`);
  }
  
  if (authContent.includes("export const useAuth")) {
    console.log(`  ✅ useAuth export found`);
  } else {
    console.log(`  ❌ useAuth export not found`);
  }
  
} catch (error) {
  console.log(`❌ AuthContext.jsx - Error: ${error.message}`);
}

console.log("\n🎯 Next steps:");
console.log("1. Start your frontend server: npm run dev");
console.log("2. Navigate to http://localhost:5174/message");
console.log("3. Login with any user account");
console.log("4. Verify that the navbar shows with role-appropriate links");
console.log("5. Test messaging functionality");

console.log("\n✨ Fix Summary:");
console.log("- Fixed AuthContext import errors");
console.log("- Added Navbar to both messaging components");
console.log("- Adjusted CSS to accommodate navbar height");
console.log("- Role-based navbar links will show based on user role");