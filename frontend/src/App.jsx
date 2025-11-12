import React, { useEffect } from 'react';
import { Routes, Route, Navigate, useLocation } from 'react-router-dom';
import './App.css';
import { AuthProvider, useAuth } from './contexts/AuthContext';
import { Login } from './components/login/Login';
import { Signup } from './components/Signup/Signup';
import EmailVerification from './components/EmailVerification/EmailVerification';
import { LandingPage } from './pages/LandingPage/LandingPage';
import { Marketplace } from './pages/Marketplace/Marketplace';
import { Collaboration } from './pages/Collaboration/Collaboration';
import StartupDetails from './pages/StartupDetails/StartupDetails';
import CreateStartupProject from './pages/CreateStartup/CreateStartupProject';
import ApplyJob from './pages/ApplyJob/ApplyJob';
import ComingSoon from './components/comingsoon/Comingsoon';
import AccountSettings from './pages/AccountSettings/AccountSettings';
import AccountSettingsII from './pages/AccountSettings/AccountSettingsII';
import Message from './pages/message/Message';
import MessageDark from './pages/message/MessageDark';
import Dashboard from './pages/Dashboard/Dashboard';
import SearchStartups from './pages/Search/SearchStartups';
import PositionManagement from './pages/PositionManagement/PositionManagement';
import PositionApplications from './pages/PositionApplications/PositionApplications';
import StartupsWithApplications from './pages/StartupsWithApplications/StartupsWithApplications';
import StartupApplications from './pages/StartupApplications/StartupApplications';
import InvestorDashboard from './pages/InvestorDashboard/InvestorDashboard';
import PitchIdea from './pages/PitchIdea/PitchIdea';
import TrendingStartups from './pages/TrendingStartups/TrendingStartups';
import ProtectedRoute from './components/ProtectedRoute/ProtectedRoute';
import RoleBasedRoute from './components/RoleBasedRoute/RoleBasedRoute';

// Component to scroll to top on route change
function ScrollToTop() {
  const { pathname } = useLocation();

  useEffect(() => {
    window.scrollTo({
      top: 0,
      left: 0,
      behavior: 'instant' // Use 'instant' for immediate scroll, or 'smooth' for animated scroll
    });
  }, [pathname]);

  return null;
}

// Main App component with routing
function AppRoutes() {
  const { isAuthenticated, user, loading, getAuthStatus } = useAuth();
  const location = useLocation();
  const [transitionStage, setTransitionStage] = React.useState('fade-in');
  
  // Debug: Log current auth status
  React.useEffect(() => {
    const status = getAuthStatus();
    console.log('🔐 Current Auth Status:', status);
  }, [isAuthenticated, user, getAuthStatus]);

  React.useEffect(() => {
    setTransitionStage('fade-out');
    const timeout = setTimeout(() => setTransitionStage('fade-in'), 180);
    return () => clearTimeout(timeout);
  }, [location]);

  if (loading) {
    return (
      <div className="spinner-overlay">
        <div className="spinner" aria-label="Loading" />
      </div>
    );
  }

  return (
    <>
      <ScrollToTop />
      <div className={`page-transition ${transitionStage}`}>
      <Routes location={location} key={location.pathname}>
      {/* Public routes */}
      <Route path="/" element={<LandingPage />} />
      <Route 
        path="/login" 
        element={isAuthenticated ? <Navigate to="/dashboard" /> : <Login />} 
      />
      <Route 
        path="/signup" 
        element={isAuthenticated ? <Navigate to="/dashboard" /> : <Signup />} 
      />
      <Route 
        path="/verify-email" 
        element={isAuthenticated ? <Navigate to="/dashboard" /> : <EmailVerification />} 
      />

      {/* Protected routes */}
      <Route path="/dashboard" element={
        <ProtectedRoute>
          <Dashboard />
        </ProtectedRoute>
      } />

      {/* Role-based routes */}
      <Route path="/marketplace" element={
        <ProtectedRoute>
          <Marketplace />
        </ProtectedRoute>
      } />
      
      <Route path="/collaboration" element={
        <ProtectedRoute>
          <Collaboration />
        </ProtectedRoute>
      } />

      <Route path="/startupdetail/:id" element={
        <ProtectedRoute>
          <StartupDetails />
        </ProtectedRoute>
      } />

      {/* Entrepreneur-only routes */}
      <Route path="/createstartup" element={
        <RoleBasedRoute allowedRoles={['entrepreneur']}>
          <CreateStartupProject />
        </RoleBasedRoute>
      } />
      
      <Route path="/pitch-idea" element={
        <RoleBasedRoute allowedRoles={['entrepreneur']}>
          <PitchIdea />
        </RoleBasedRoute>
      } />

      {/* Search routes - accessible to all authenticated users */}
      <Route path="/search" element={
        <ProtectedRoute>
          <SearchStartups />
        </ProtectedRoute>
      } />
      
      {/* Trending Startups - accessible to all authenticated users */}
      <Route path="/trending-startups" element={
        <ProtectedRoute>
          <TrendingStartups />
        </ProtectedRoute>
      } />
      
      {/* Entrepreneur-only routes */}
      <Route path="/startups/:startupId/positions" element={
        <RoleBasedRoute allowedRoles={['entrepreneur']}>
          <PositionManagement />
        </RoleBasedRoute>
      } />
      
      <Route path="/positions/:positionId/applications" element={
        <RoleBasedRoute allowedRoles={['entrepreneur']}>
          <PositionApplications />
        </RoleBasedRoute>
      } />
      
      <Route path="/startups-with-applications" element={
        <RoleBasedRoute allowedRoles={['entrepreneur']}>
          <StartupsWithApplications />
        </RoleBasedRoute>
      } />
      
      <Route path="/startups/:startupId/applications" element={
        <RoleBasedRoute allowedRoles={['entrepreneur']}>
          <StartupApplications />
        </RoleBasedRoute>
      } />
      
      {/* Investor-only routes */}
      <Route path="/investor-dashboard" element={
        <RoleBasedRoute allowedRoles={['investor']}>
          <InvestorDashboard />
        </RoleBasedRoute>
      } />

      {/* Student/Professional and Entrepreneur routes */}
      <Route path="/apply-for-collaboration/:startupId" element={
        <RoleBasedRoute allowedRoles={['student', 'entrepreneur']}>
          <ApplyJob />
        </RoleBasedRoute>
      } />

      {/* General protected routes */}
      <Route path="/account" element={
        <ProtectedRoute>
          <AccountSettingsII />
        </ProtectedRoute>
      } />

      <Route path="/message" element={
        <ProtectedRoute>
          <MessageDark />
        </ProtectedRoute>
      } />

      {/* Catch all route */}
      <Route path="*" element={<Navigate to="/" />} />
      </Routes>
      </div>
    </>
  );
}

function App() {
  return (
    <AuthProvider>
      <AppRoutes />
    </AuthProvider>
  );
}

export default App;
