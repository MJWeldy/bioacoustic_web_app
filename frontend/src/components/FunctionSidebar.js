import React from 'react';
import './FunctionSidebar.css';

const FunctionSidebar = ({ selectedFunction, onFunctionSelect }) => {
  const functions = [
    {
      id: 'active_learning',
      icon: '🎯',
      label: 'Active Learning',
      description: 'Build datasets and train models iteratively'
    },
    {
      id: 'validation',
      icon: '✅',
      label: 'Validation',
      description: 'Test and validate model performance'
    }
    // Call Density feature removed - under development
  ];

  return (
    <div className="function-sidebar">
      <div className="function-icons">
        {functions.map((func) => (
          <div
            key={func.id}
            className={`function-icon ${selectedFunction === func.id ? 'active' : ''}`}
            onClick={() => onFunctionSelect(func.id)}
            title={func.description}
          >
            <div className="icon">{func.icon}</div>
            <div className="label">{func.label}</div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default FunctionSidebar;