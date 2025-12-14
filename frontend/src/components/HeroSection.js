import React from 'react';
import '../styles/HeroSection.css';

const HeroSection = () => {
  return (
    <div className="hero-container">
      <header className="hero-header">
        <h1>AI LAWYER ASSISTANT</h1>
        <h2>LEXISIGHT</h2>
        <button className="get-started-btn">
          <a href="/signup">Get Started</a>
        </button>
        <p>
        This project proposes developing an AI-powered personal lawyer assistant to democratize access to legal advice, which is often costly and time-intensive. By leveraging technologies like XLNet, RAG, and vector databases, this AI model will provide users with relevant legal information, insightful sections of the law, and strategic guidance across diverse legal domains, effectively simulating the expertise of a legal professional.
        </p>
      </header>
    </div>
  );
};

export default HeroSection;
