import React from 'react';
import '../styles/Login.css';

const Login = () => {
  return (
    <div className="login-container">
      <div className="login-form">
        <h2>Welcome back!</h2>
        <form>
          <div className="form-group">
            <label>Email address</label>
            <input type="email" placeholder="Enter your email" />
          </div>
          <div className="form-group">
            <label>Password</label>
            <input type="password" placeholder="Enter password" />
            <a href="#forgot-password" className="forgot-password">Forgot password?</a>
          </div>
          <div className="form-group checkbox-group">
            <input type="checkbox" id="remember" />
            <label htmlFor="remember">Remember for 30 days</label>
          </div>
          <button type="submit" className="login-button">Login</button>
        </form>
        <p>Don't have an account? <a href="/signup">Sign Up</a></p>
      </div>
      <div className="login-image">
        {/* <img src="/ai-law.webp" alt="AI Law Illustration" style={{ width: '100%', maxWidth: '600px' }} /> */}
      </div>
    </div>
  );
};

export default Login;
