import React, { useState } from 'react';

const TestForm = () => {
  const [formData, setFormData] = useState({
    title: '',
    description: '',
    field: ''
  });

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    console.log('🔥 TEST FORM SUBMITTED!');
    console.log('Form data:', formData);
    alert('Form submitted! Check console for data.');
  };

  const handleButtonClick = (e) => {
    console.log('🚨 TEST BUTTON CLICKED!', e.type);
    console.log('Event target:', e.target);
  };

  return (
    <div style={{ padding: '20px', maxWidth: '500px' }}>
      <h2>Test Form Component</h2>
      <form onSubmit={handleSubmit}>
        <div style={{ marginBottom: '15px' }}>
          <label>Title:</label>
          <input 
            type="text" 
            name="title"
            value={formData.title}
            onChange={handleInputChange}
            placeholder="Enter title"
            style={{ display: 'block', width: '100%', padding: '5px' }}
          />
        </div>
        
        <div style={{ marginBottom: '15px' }}>
          <label>Description:</label>
          <textarea 
            name="description"
            value={formData.description}
            onChange={handleInputChange}
            placeholder="Enter description"
            rows="4"
            style={{ display: 'block', width: '100%', padding: '5px' }}
          />
        </div>
        
        <div style={{ marginBottom: '15px' }}>
          <label>Field:</label>
          <input 
            type="text" 
            name="field"
            value={formData.field}
            onChange={handleInputChange}
            placeholder="Enter field"
            style={{ display: 'block', width: '100%', padding: '5px' }}
          />
        </div>
        
        <div>
          <button 
            type="submit"
            onClick={handleButtonClick}
            style={{ 
              padding: '10px 20px', 
              backgroundColor: '#007bff', 
              color: 'white', 
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer'
            }}
          >
            Test Submit
          </button>
        </div>
      </form>
    </div>
  );
};

export default TestForm;