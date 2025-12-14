import React from 'react';
import '../styles/Signup.css';

const Signup = () => {
  return (
    <div className="signup-container">
      <div className="signup-form">
        <h2>Get Started Now</h2>
        <form>
          <div className="form-group">
            <label>Name</label>
            <input type="text" placeholder="Enter your name" />
          </div>
          <div className="form-group">
            <label>Email address</label>
            <input type="email" placeholder="Enter your email" />
          </div>
          <div className="form-group">
            <label>Password</label>
            <input type="password" placeholder="Enter password" />
          </div>
          <div className="form-group checkbox-group">
            <input type="checkbox" id="terms" />
            <label htmlFor="terms">I agree to all terms & policy</label>
          </div>
          <button type="submit" className="signup-button">Signup</button>
        </form>
        <p>Have an account? <a href="/login">Sign In</a></p>
      </div>
      <div className="signup-image">
        {/* <img src="/ai-law.webp" alt="AI Law Illustration" style={{ width: '100%', maxWidth: '600px' }} /> */}
      </div>
    </div>
  );
};

export default Signup;
