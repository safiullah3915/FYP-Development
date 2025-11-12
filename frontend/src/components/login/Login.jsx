import React, { useState, useEffect } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import "./login.css";
import { Submit } from "../Submit/Submit";
import { Bounce, ToastContainer, toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import { useAuth } from "../../contexts/AuthContext";
import { Navbar } from "../Navbar/Navbar";


function Login({leftimg, logo, line1, line2, paragraph}){
    const { login, isAuthenticated } = useAuth();
    const navigate = useNavigate();
    const [searchParams] = useSearchParams();
    
    //javascript-------------------------------------------------------------
    const [password, setPassword] = useState("")
    const [email, setEmail] = useState("")
    const [show, setShow] = useState(false);
    const [loading, setLoading] = useState(false);
    
    // Redirect if already authenticated
    useEffect(() => {
        if (isAuthenticated) {
            navigate("/dashboard");
        }
    }, [isAuthenticated, navigate]);
    
    // Pre-fill email from URL params if available
    useEffect(() => {
        const emailParam = searchParams.get('email');
        if (emailParam) {
            setEmail(emailParam);
        }
    }, [searchParams]);
    
    const forgetPasswords = ()=>{
        navigate("/auth/forget-password")
    }
    
    const passwordToggle = ()=>{
        setShow(!show)
    }
    
    //Notifications
    
    function passwordError(){
        toast.error('Password must be greater than 8 characters', {
            position: "top-center",
            autoClose: 3000,
            hideProgressBar: false,
            closeOnClick: true,
            pauseOnHover: true,
            draggable: true,
            theme: "light",
            transition: Bounce,
            });
    }
    function wrongCredentials(){
        toast.error('Wrong credentials', {
            position: "top-center",
            autoClose: 3000,
            hideProgressBar: false,
            closeOnClick: true,
            pauseOnHover: true,
            draggable: true,
            theme: "light",
            transition: Bounce,
            });
    }
    function emailNotVerified(){
        toast.error('Email not verified. Please check your email for verification code.', {
            position: "top-center",
            autoClose: 5000,
            hideProgressBar: false,
            closeOnClick: true,
            pauseOnHover: true,
            draggable: true,
            theme: "light",
            transition: Bounce,
            });
    }
    
    function serverError(){
        toast.error('Internal Server Error', {
            position: "top-center",
            autoClose: 3000,
            hideProgressBar: false,
            closeOnClick: true,
            pauseOnHover: true,
            draggable: true,
            theme: "light",
            transition: Bounce,
            });
    }
    
    
    function success() {
        toast.success('Login Succesfully', {
            position: "top-center",
            autoClose: 3000,
            hideProgressBar: false,
            closeOnClick: true,
            pauseOnHover: true,
            draggable: true,
            theme: "light",
            transition: Bounce,
            });
    };
    
    
    
    
    
    const loginCredentials = async (e) => {
        e.preventDefault();
        
        if (!email.trim() || !password.trim()) {
            toast.error('Please fill in all fields');
            return;
        }
        
        setLoading(true);
        
        try {
            const result = await login(email, password);
            
            if (result.success) {
                success();
                navigate("/dashboard");
            } else {
                if (result.error === "Wrong credentials") {
                    wrongCredentials();
                } else if (result.needsVerification || result.error === "Email not verified" || result.error.includes("not verified")) {
                    emailNotVerified();
                    // Store email for verification and redirect
                    localStorage.setItem('pendingLoginEmail', email);
                    setTimeout(() => {
                        navigate(`/verify-email?email=${encodeURIComponent(email)}`);
                    }, 2000);
                } else {
                    serverError();
                }
            }
        } catch (error) {
            console.error("Login failed:", error);
            serverError();
        } finally {
            setLoading(false);
        }
    }


    // Remove old cookie-based authentication code



  
    
    let allFieldsFill = (e)=>{
        return  password.trim() !== "" && email.trim() !== "";
    }
    //-----------------------------------------------------------------------------------
    
    
    
       
    //call
    return(
        <>
        <Navbar />
        {loading && (
            <div className="progress-overlay">
                <div className="progress-modal">
                    <div className="progress-spinner-large"></div>
                    <p className="progress-text">Signing In...</p>
                </div>
            </div>
        )}
        <div className="main signin-page">
    
    <div className="left">
    <div className="leftimg"><img src="./login-banner.png" alt="Login Banner" /></div>
    </div>
    
    
        <div className="right">
    <div className="loginbox">
    <div className="logo"></div>
    
    <h2 className="welcome-title">Welcome Back</h2>
    <p className="welcome-subtitle">Sign in to continue your entrepreneurial journey</p>
    
        <label htmlFor="email" >Email Address</label>
        <input type="email" name = "email" placeholder="Enter your email address" className="inp" value={email} onChange={(e)=>{setEmail(e.target.value)}}/>
    
    
        <label htmlFor="password" >Password</label>
        <div className="pas">
    
        <input type={show?"text" : "password"} name="password" placeholder="Enter your password" className="inp" onChange={(e)=>{setPassword(e.target.value)}}/>
        <p onClick={passwordToggle}>{show?"hide":"show"}</p>
        </div>
        <p className="forgot" onClick={forgetPasswords}>Forgot Password?</p>

            <form method = 'POST'>
            <input 
                type="submit" 
                disabled={!allFieldsFill() || loading} 
                onClick={loginCredentials} 
            value="Sign In" 
                className="btn"
            />
            
            </form>

        <Link to="/signup" className="noaccount">
        <p className="signupText">Don't have an account? <span className="signup-link">Sign up here</span></p>
        </Link>
    </div>
        </div>
        </div>
        <ToastContainer/>
        </>
    );
    }
    
    export {Login}






// import React from 'react'

// function Login() {
//   return (
//     <div>
//       hiii
//     </div>
//   )
// }

// export {Login}
