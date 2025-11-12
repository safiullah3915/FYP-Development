import React, { useState, useEffect, useCallback } from 'react';
import { useAuth } from '../../contexts/AuthContext';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { Bounce, ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import "./EmailVerification.css";

const EmailVerification = () => {
  const [loading, setLoading] = useState(false);
  const [email, setEmail] = useState('');
  const [verificationCode, setVerificationCode] = useState('');
  const { verifyEmail, requestNewVerificationToken } = useAuth();
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();

  const handleVerification = useCallback(async (e) => {
    if (e) e.preventDefault();
    
    if (!verificationCode.trim()) {
      toast.error('Please enter the verification code');
      return;
    }

    setLoading(true);

    try {
      // Send the user-entered verification code
      const result = await verifyEmail(verificationCode, email);
      
      if (result.success) {
        toast.success('Email verified successfully!');
        
        // Clear verification code state
        setVerificationCode('');
        
        // Try to get stored email for login redirect
        const pendingEmail = localStorage.getItem('pendingLoginEmail');
        
        if (pendingEmail) {
          toast.info('You can now login with your credentials.');
          // Clear stored email and redirect to login
          localStorage.removeItem('pendingLoginEmail');
          setTimeout(() => {
            navigate(`/login?email=${encodeURIComponent(pendingEmail)}`);
          }, 2000);
        } else {
          setTimeout(() => {
            navigate('/login');
          }, 2000);
        }
      } else {
        toast.error(result.error || 'Verification failed');
      }
    } catch (error) {
      toast.error('Verification failed');
    } finally {
      setLoading(false);
    }
  }, [verificationCode, email, verifyEmail, navigate]);

  const sendVerificationCode = async (emailAddress) => {
    try {
      console.log('Sending verification code to:', emailAddress);
      const result = await requestNewVerificationToken(emailAddress);
      
      if (result.success) {
        toast.success('Verification code sent to your email!');
      } else {
        toast.error(result.error || 'Failed to send verification code');
      }
    } catch (error) {
      console.error('Failed to send verification code:', error);
      toast.error('Failed to send verification code');
    }
  };
  
  const resendVerificationCode = async () => {
    if (!email) {
      toast.error('Email address not available');
      return;
    }
    
    setLoading(true);
    try {
      await sendVerificationCode(email);
      toast.info('Verification code resent!');
    } catch (error) {
      toast.error('Failed to resend code');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    // Get email from URL params if available
    const emailParam = searchParams.get('email');
    if (emailParam) {
      setEmail(emailParam);
    }
    
    // Get email from localStorage if available
    const pendingEmail = localStorage.getItem('pendingLoginEmail');
    if (pendingEmail && !emailParam) {
      setEmail(pendingEmail);
    }
    
    // Don't automatically send verification code - it was already sent during signup
    // The signup process already sends the verification code, so we just need to display the form
  }, [searchParams, navigate]);


  return (
    <div className="verification-container">
      <div className="verification-box">
        <div className="logo-section">
          <img src="../../../images/logolight.png" alt="Logo" className="logo" />
        </div>
        
        <div className="verification-content">
          <h1>Verify Your Email</h1>
          <p className="verification-text">
            We've sent a verification code to {email}. Please enter the 6-digit code below to verify your account.
          </p>

          <form onSubmit={handleVerification}>
            <div className="input-group">
              <label htmlFor="verification-code">Verification Code</label>
              <input
                type="text"
                id="verification-code"
                placeholder="Enter 6-digit code"
                value={verificationCode}
                onChange={(e) => setVerificationCode(e.target.value)}
                className="verification-input"
                maxLength="6"
                required
              />
            </div>

            <button 
              type="submit" 
              disabled={loading || !verificationCode.trim()}
              className="verify-button"
            >
              {loading ? 'Verifying...' : 'Verify Email'}
            </button>
          </form>

          <div className="verification-actions">
            <p>Didn't receive the code?</p>
            <button 
              type="button" 
              onClick={resendVerificationCode}
              className="resend-button"
              disabled={loading}
            >
              Resend Code
            </button>
          </div>

          <div className="back-to-login">
            <button 
              onClick={() => navigate('/login')}
              className="back-button"
            >
              Back to Login
            </button>
          </div>
        </div>
      </div>
      <ToastContainer />
    </div>
  );
};

export default EmailVerification;