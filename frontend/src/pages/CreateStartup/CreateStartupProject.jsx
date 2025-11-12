import React, { useState } from "react";
import styles from "./CreateStartupProject.module.css"; // Use CSS Module
import { Navbar } from '../../components/Navbar/Navbar';
import { Footer } from '../../components/Footer/Footer';
import { useNavigate } from 'react-router-dom';
import { toast } from 'react-toastify';
import { useAuth } from '../../contexts/AuthContext';
import apiClient from '../../utils/axiosConfig';
import { debugAuthStatus } from '../../utils/authDebug';

const CreateStartupProject = () => {
  const navigate = useNavigate();
  const { user } = useAuth();
  const [loading, setLoading] = useState(false);
  const DEBUG_SKIP_VALIDATION = true; // Set to false to enable validation
  const [formData, setFormData] = useState({
    title: '',
    role_title: '',
    description: '',
    field: '',
    website_url: '',
    stages: [],
    stage: '',
    revenue: '',
    profit: '',
    asking_price: '',
    ttm_revenue: '',
    ttm_profit: '',
    last_month_revenue: '',
    last_month_profit: '',
    type: 'marketplace',
    earn_through: '',
    phase: '',
    team_size: '',
    category: 'saas'
  });

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleCheckboxChange = (e) => {
    const { name, checked } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: checked ? [...prev[name], e.target.value] : prev[name].filter(item => item !== e.target.value)
    }));
  };

  const handleButtonClick = (e) => {
    console.log('🚨 Button clicked!', e.type);
    console.log('🎯 Event target:', e.target);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    console.log('🔥 Form submitted! Handler called');
    console.log('📊 Current form data:', formData);
    
    if (DEBUG_SKIP_VALIDATION) {
      console.log('🚀 DEBUG MODE: Skipping validation - proceeding directly to API call');
    } else {
      // Validate required fields according to backend model
      console.log('🔍 Starting validation...');
      console.log('Title validation - Value:', formData.title, 'Length:', formData.title?.length);
      if (!formData.title || formData.title.length < 3) {
        console.log('❌ Title validation FAILED');
        toast.error('Startup name must be at least 3 characters long');
        return;
      }
      console.log('✅ Title validation PASSED');
      
      console.log('Description validation - Value:', formData.description, 'Length:', formData.description?.length);
      if (!formData.description || formData.description.length < 5) {
        console.log('❌ Description validation FAILED');
        toast.error('Description must be at least 5 characters long');
        return;
      }
      console.log('✅ Description validation PASSED');
      
      console.log('Field validation - Value:', formData.field, 'Trimmed:', formData.field?.trim());
      if (!formData.field || formData.field.trim() === '') {
        console.log('❌ Field validation FAILED');
        toast.error('Field/Industry is required');
        return;
      }
      console.log('✅ Field validation PASSED');
    }

    console.log('✅ All validations passed, proceeding with API call...');
    console.log('👤 Current user:', user);
    console.log('🔐 Authentication status:', user ? 'Authenticated' : 'Not authenticated');
    
    // Simple authentication check using AuthContext
    if (!user || !user.id) {
      console.error('❌ AUTHENTICATION FAILED: User not logged in!');
      toast.error('Please login first to create a startup');
      navigate('/login');
      return;
    }
    
    console.log('✅ User authenticated:', user.username);
    
    // Run debug for troubleshooting (optional)
    debugAuthStatus();
    
    setLoading(true);
    
    try {
      console.log('🚀 ================ STARTUP CREATION ATTEMPT ================');
      console.log('📋 Form data being sent:', formData);
      console.log('🚀 Making API call to /api/startups...');
      
      // Ensure data types match backend expectations
      const cleanedFormData = {
        title: formData.title,
        role_title: formData.role_title,
        description: formData.description,
        field: formData.field,
        website_url: formData.website_url || '',
        // Convert single stage to array for backend (backend expects ListField)
        stages: formData.stage ? [formData.stage] : [],
        revenue: formData.revenue || '',
        profit: formData.profit || '',
        asking_price: formData.asking_price || '',
        ttm_revenue: formData.ttm_revenue || '',
        ttm_profit: formData.ttm_profit || '',
        last_month_revenue: formData.last_month_revenue || '',
        last_month_profit: formData.last_month_profit || '',
        type: formData.type,
        earn_through: formData.earn_through || '',
        phase: formData.phase || '',
        team_size: formData.team_size || '',
        category: formData.category || 'other'
      };
      
      console.log('🧺 Cleaned form data:', cleanedFormData);
      console.log('🌐 About to make API call to /api/startups...');
      
      const response = await apiClient.post('/api/startups', cleanedFormData);
      
      console.log('✅ API call successful!', response);
      console.log('📄 Response data:', response.data);
      
      toast.success('Startup created successfully!');
      navigate('/dashboard');
    } catch (error) {
      console.error('❌ DETAILED ERROR ANALYSIS:');
      console.error('Full error object:', error);
      console.error('Error message:', error.message);
      console.error('Error response:', error.response);
      console.error('Error response data:', error.response?.data);
      console.error('Error response status:', error.response?.status);
      console.error('Error response headers:', error.response?.headers);
      console.error('Request config:', error.config);
      console.error('Request URL:', error.config?.url);
      console.error('Request method:', error.config?.method);
      console.error('Request headers:', error.config?.headers);
      // Use the formData here since cleanedFormData is only defined in the try block
      console.error('Form data that was sent:', formData);
      
      // Check if this is a network error
      if (!error.response) {
        console.error('🌐 NETWORK ERROR: No response received from server');
        console.error('Possible causes:');
        console.error('- Backend server is not running');
        console.error('- Wrong API URL');
        console.error('- CORS issues');
        console.error('- Network connectivity problems');
      }
      
      // Better error message parsing
      let errorMessage = 'Failed to create startup. Please try again.';
      
      if (error.response?.data) {
        const data = error.response.data;
        if (typeof data === 'string' && data.includes('AnonymousUser')) {
          errorMessage = 'Authentication failed. Please log in and try again.';
        } else if (data.message) {
          errorMessage = data.message;
        } else if (data.error) {
          errorMessage = data.error;
        } else if (data.detail) {
          errorMessage = data.detail;
        } else if (typeof data === 'string' && data.length < 200) {
          errorMessage = data;
        }
      } else if (error.message) {
        errorMessage = error.message;
      }
      
      console.error('💬 Error message to user:', errorMessage);
      toast.error(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
    <Navbar/>
    <div className={styles.container}>
      <form className={styles.form} onSubmit={handleSubmit}>
        <h2>Create Startup Project</h2>

          <div className={styles.formGroup}>
            <label>Startup name *</label>
            <input 
              type="text" 
              name="title"
              value={formData.title}
              onChange={handleInputChange}
              placeholder="Type your startup name" 
              required
            />
        </div>


        <div className={styles.formGroup}>
          <label>Categories</label>
          <div className={styles.category}>
            <label className={`${styles.catg} ${formData.category === 'saas' ? styles.selected : ''}`}>
              <input 
                type="radio" 
                name="category" 
                value="saas" 
                checked={formData.category === 'saas'}
                onChange={handleInputChange}
              />
              <span>SaaS</span>
            </label>
            <label className={`${styles.catg} ${formData.category === 'ecommerce' ? styles.selected : ''}`}>
              <input 
                type="radio" 
                name="category" 
                value="ecommerce" 
                checked={formData.category === 'ecommerce'}
                onChange={handleInputChange}
              />
              <span>Ecommerce</span>
            </label>
            <label className={`${styles.catg} ${formData.category === 'agency' ? styles.selected : ''}`}>
              <input 
                type="radio" 
                name="category" 
                value="agency" 
                checked={formData.category === 'agency'}
                onChange={handleInputChange}
              />
              <span>Agency</span>
            </label>
            <label className={`${styles.catg} ${formData.category === 'legal' ? styles.selected : ''}`}>
              <input 
                type="radio" 
                name="category" 
                value="legal" 
                checked={formData.category === 'legal'}
                onChange={handleInputChange}
              />
              <span>Legal</span>
            </label>
            <label className={`${styles.catg} ${formData.category === 'marketplace' ? styles.selected : ''}`}>
              <input 
                type="radio" 
                name="category" 
                value="marketplace" 
                checked={formData.category === 'marketplace'}
                onChange={handleInputChange}
              />
              <span>Marketplace</span>
            </label>
            <label className={`${styles.catg} ${formData.category === 'media' ? styles.selected : ''}`}>
              <input 
                type="radio" 
                name="category" 
                value="media" 
                checked={formData.category === 'media'}
                onChange={handleInputChange}
              />
              <span>Media</span>
            </label>
            <label className={`${styles.catg} ${formData.category === 'platform' ? styles.selected : ''}`}>
              <input 
                type="radio" 
                name="category" 
                value="platform" 
                checked={formData.category === 'platform'}
                onChange={handleInputChange}
              />
              <span>Platform</span>
            </label>
            <label className={`${styles.catg} ${formData.category === 'real_estate' ? styles.selected : ''}`}>
              <input 
                type="radio" 
                name="category" 
                value="real_estate" 
                checked={formData.category === 'real_estate'}
                onChange={handleInputChange}
              />
              <span>Real Estate</span>
            </label>
            <label className={`${styles.catg} ${formData.category === 'robotics' ? styles.selected : ''}`}>
              <input 
                type="radio" 
                name="category" 
                value="robotics" 
                checked={formData.category === 'robotics'}
                onChange={handleInputChange}
              />
              <span>Robotics</span>
            </label>
            <label className={`${styles.catg} ${formData.category === 'software' ? styles.selected : ''}`}>
              <input 
                type="radio" 
                name="category" 
                value="software" 
                checked={formData.category === 'software'}
                onChange={handleInputChange}
              />
              <span>Software</span>
            </label>
            <label className={`${styles.catg} ${formData.category === 'web3' ? styles.selected : ''}`}>
              <input 
                type="radio" 
                name="category" 
                value="web3" 
                checked={formData.category === 'web3'}
                onChange={handleInputChange}
              />
              <span>Web3</span>
            </label>
            <label className={`${styles.catg} ${formData.category === 'crypto' ? styles.selected : ''}`}>
              <input 
                type="radio" 
                name="category" 
                value="crypto" 
                checked={formData.category === 'crypto'}
                onChange={handleInputChange}
              />
              <span>Crypto</span>
            </label>
            <label className={`${styles.catg} ${formData.category === 'other' ? styles.selected : ''}`}>
              <input 
                type="radio" 
                name="category" 
                value="other" 
                checked={formData.category === 'other'}
                onChange={handleInputChange}
              />
              <span>Other</span>
            </label>
          </div>
          
          <label>Startup Description *</label>
          <textarea 
            name="description"
            value={formData.description}
            onChange={handleInputChange}
            placeholder="Write a description here" 
            rows="4"
            required
          ></textarea>
          
          <label>Field/Industry *</label>
          <input 
            type="text" 
            name="field"
            value={formData.field}
            onChange={handleInputChange}
            placeholder="e.g., Technology, Healthcare, Finance" 
            required
          />
        </div>

        <div className={styles.formGroup}>
          <label>Stage of your startup</label>
          <div className={styles.checkboxGroup}>
            <label className={styles.checkboxlabel}>
              <input 
                type="radio" 
                name="stage"
                value="Idea Stage"
                checked={formData.stage === 'Idea Stage'}
                onChange={handleInputChange}
              /> Idea Stage
            </label>
            <label className={styles.checkboxlabel}>
              <input 
                type="radio" 
                name="stage"
                value="Building MVP"
                checked={formData.stage === 'Building MVP'}
                onChange={handleInputChange}
              /> Building MVP
            </label>
            <label className={styles.checkboxlabel}>
              <input 
                type="radio" 
                name="stage"
                value="MVP Stage"
                checked={formData.stage === 'MVP Stage'}
                onChange={handleInputChange}
              /> MVP Stage
            </label>
            <label className={styles.checkboxlabel}>
              <input 
                type="radio" 
                name="stage"
                value="Product Market Fit"
                checked={formData.stage === 'Product Market Fit'}
                onChange={handleInputChange}
              /> Product Market Fit
            </label>
            <label className={styles.checkboxlabel}>
              <input 
                type="radio" 
                name="stage"
                value="Fund raising"
                checked={formData.stage === 'Fund raising'}
                onChange={handleInputChange}
              /> Fund raising
            </label>
            <label className={styles.checkboxlabel}>
              <input 
                type="radio" 
                name="stage"
                value="Growth"
                checked={formData.stage === 'Growth'}
                onChange={handleInputChange}
              /> Growth
            </label>
            <label className={styles.checkboxlabel}>
              <input 
                type="radio" 
                name="stage"
                value="Exit"
                checked={formData.stage === 'Exit'}
                onChange={handleInputChange}
              /> Exit
            </label>
          </div>
        </div>

        <div className={styles.formGroup}>
          <label>Startup Type</label>
          <select 
            name="type" 
            value={formData.type} 
            onChange={handleInputChange}
          >
            <option value="marketplace">Marketplace (For Sale)</option>
            <option value="collaboration">Collaboration (Looking for Team)</option>
          </select>
        </div>

        {/* Conditional fields based on type */}
        {formData.type === 'marketplace' && (
          <>
            {/*<h3>Marketplace Information (For Sale)</h3>*/}
            
            <div className={styles.row}>
              <div className={styles.formGroup}>
                <label>Website URL</label>
                <input 
                  type="url" 
                  name="website_url"
                  value={formData.website_url}
                  onChange={handleInputChange}
                  placeholder="https://demowebsite.com" 
                  required
                />
              </div>
              <div className={styles.formGroup}>
                <label>Current Phase</label>
                <input 
                  type="text" 
                  name="phase"
                  value={formData.phase}
                  onChange={handleInputChange}
                  placeholder="e.g., Seed Stage, Series A" 
                  required
                />
              </div>
            </div>
            
            <div className={styles.row}>
              <div className={styles.formGroup}>
                <label>Team Size</label>
                <input 
                  type="text" 
                  name="team_size"
                  value={formData.team_size}
                  onChange={handleInputChange}
                  placeholder="e.g., 2-5 people" 
                  required
                />
              </div>
              <div className={styles.formGroup}>
                <label>How do you earn?</label>
                <input 
                  type="text" 
                  name="earn_through"
                  value={formData.earn_through}
                  onChange={handleInputChange}
                  placeholder="e.g., Subscriptions, Sales, Ads" 
                  required
                />
              </div>
            </div>
            
            {/*<h4>Financial Information *</h4>*/}
            <div className={styles.row}>
              <div className={styles.formGroup}>
                <label>Current Revenue *</label>
                <input 
                  type="text" 
                  name="revenue"
                  value={formData.revenue}
                  onChange={handleInputChange}
                  placeholder="e.g., $10,000/month" 
                  required
                />
              </div>
              <div className={styles.formGroup}>
                <label>Current Profit *</label>
                <input 
                  type="text" 
                  name="profit"
                  value={formData.profit}
                  onChange={handleInputChange}
                  placeholder="e.g., $5,000/month" 
                  required
                />
              </div>
            </div>
            
            <div className={styles.row}>
              <div className={styles.formGroup}>
                <label>Asking Price *</label>
                <input 
                  type="text" 
                  name="asking_price"
                  value={formData.asking_price}
                  onChange={handleInputChange}
                  placeholder="e.g., $100,000" 
                  required
                />
              </div>
              <div className={styles.formGroup}>
                <label>TTM Revenue *</label>
                <input 
                  type="text" 
                  name="ttm_revenue"
                  value={formData.ttm_revenue}
                  onChange={handleInputChange}
                  placeholder="e.g., $120,000" 
                  required
                />
              </div>
            </div>
            
            <div className={styles.row}>
              <div className={styles.formGroup}>
                <label>TTM Profit *</label>
                <input 
                  type="text" 
                  name="ttm_profit"
                  value={formData.ttm_profit}
                  onChange={handleInputChange}
                  placeholder="e.g., $60,000" 
                  required
                />
              </div>
              <div className={styles.formGroup}>
                <label>Last Month Revenue *</label>
                <input 
                  type="text" 
                  name="last_month_revenue"
                  value={formData.last_month_revenue}
                  onChange={handleInputChange}
                  placeholder="e.g., $12,000" 
                  required
                />
              </div>
            </div>
            
            <div className={styles.formGroup}>
              <label>Last Month Profit *</label>
              <input 
                type="text" 
                name="last_month_profit"
                value={formData.last_month_profit}
                onChange={handleInputChange}
                placeholder="e.g., $6,000" 
                required
              />
            </div>
          </>
        )}

        {formData.type === 'collaboration' && (
          <>
            {/*<h3>Collaboration Details (Looking for Team)</h3>*/}
            
            <div className={styles.row}>
              <div className={styles.formGroup}>
                <label>Website URL</label>
                <input 
                  type="url" 
                  name="website_url"
                  value={formData.website_url}
                  onChange={handleInputChange}
                  placeholder="https://demowebsite.com" 
                  required
                />
              </div>
              <div className={styles.formGroup}>
                <label>Current Phase</label>
                <input 
                  type="text" 
                  name="phase"
                  value={formData.phase}
                  onChange={handleInputChange}
                  placeholder="e.g., Seed Stage, Series A, Idea Stage" 
                  required
                />
              </div>
            </div>
            
            <div className={styles.row}>
              <div className={styles.formGroup}>
                <label>Current Team Size</label>
                <input 
                  type="text" 
                  name="team_size"
                  value={formData.team_size}
                  onChange={handleInputChange}
                  placeholder="e.g., 1-2 people, Just me" 
                  required
                />
              </div>
              <div className={styles.formGroup}>
                <label>How will team members earn?</label>
                <input 
                  type="text" 
                  name="earn_through"
                  value={formData.earn_through}
                  onChange={handleInputChange}
                  placeholder="e.g., Equity, Revenue Share, Salary" 
                  required
                />
              </div>
            </div>
          </>
        )}

        <div className={styles.actionButtons}>
          <button 
            type="button" 
            className={styles.cancelBtn}
            onClick={() => navigate('/dashboard')}
          >
            Cancel
          </button>
          <button 
            type="submit" 
            className={styles.submitBtn}
            disabled={loading}
            onClick={handleButtonClick}
          >
            {loading ? 'Creating...' : 'Create Project'}
          </button>
        </div>
      </form>
    </div>

<Footer/>
    </>
  );
};

export default CreateStartupProject;
