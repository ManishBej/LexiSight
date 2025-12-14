import React from 'react';
import { Link } from 'react-router-dom';
import '../styles/Header.css';

const Header = () => {
  return (
    <header className="header">
      <div className="logo">
      <Link to="/">LexiSight</Link></div>
      <nav>
        <ul>
          <li><a href="#">Services</a></li>
          <li><a href="#">About Project</a></li>
          <li><a href="#">Team</a></li>
        </ul>
      </nav>
    </header>
  );
};

export default Header;
