import React from 'react';
import './BouncingDots.css';

const BouncingDots = () => {
  return (
    <div className="bouncing-loader">
      <div className="dot"></div>
      <div className="dot"></div>
      <div className="dot"></div>
    </div>
  );
};

export default BouncingDots;