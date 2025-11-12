import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { recommendationAPI } from '../../utils/apiServices';
import styles from './OnboardingPreferences.module.css';

const CATEGORIES = [
  'saas', 'ecommerce', 'agency', 'legal', 'marketplace', 
  'media', 'platform', 'real_estate', 'robotics', 
  'software', 'web3', 'crypto', 'other'
];

const ENGAGEMENT_TYPES = ['full-time', 'part-time', 'equity', 'paid'];

const OnboardingPreferences = () => {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [preferences, setPreferences] = useState({
    selected_categories: [],
    selected_fields: [],
    selected_tags: [],
    preferred_startup_stages: [],
    preferred_engagement_types: [],
    preferred_skills: [],
  });
  const [newField, setNewField] = useState('');
  const [newTag, setNewTag] = useState('');
  const [newStage, setNewStage] = useState('');
  const [newSkill, setNewSkill] = useState('');

  useEffect(() => {
    // Load existing preferences if any
    recommendationAPI.getOnboardingPreferences()
      .then(response => {
        if (response.data) {
          setPreferences({
            selected_categories: response.data.selected_categories || [],
            selected_fields: response.data.selected_fields || [],
            selected_tags: response.data.selected_tags || [],
            preferred_startup_stages: response.data.preferred_startup_stages || [],
            preferred_engagement_types: response.data.preferred_engagement_types || [],
            preferred_skills: response.data.preferred_skills || [],
          });
        }
      })
      .catch(error => {
        // If no preferences exist, that's okay
        console.log('No existing preferences found');
      });
  }, []);

  const toggleSelection = (field, value) => {
    setPreferences(prev => {
      const current = prev[field] || [];
      const updated = current.includes(value)
        ? current.filter(item => item !== value)
        : [...current, value];
      return { ...prev, [field]: updated };
    });
  };

  const addCustomItem = (field, value, setter) => {
    if (value.trim() && !preferences[field].includes(value.trim())) {
      setPreferences(prev => ({
        ...prev,
        [field]: [...prev[field], value.trim()]
      }));
      setter('');
    }
  };

  const removeItem = (field, value) => {
    setPreferences(prev => ({
      ...prev,
      [field]: prev[field].filter(item => item !== value)
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    
    try {
      await recommendationAPI.saveOnboardingPreferences({
        ...preferences,
        onboarding_completed: true
      });
      navigate('/dashboard'); // Redirect to dashboard after completion
    } catch (error) {
      console.error('Error saving preferences:', error);
      alert('Failed to save preferences. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleSkip = () => {
    navigate('/dashboard');
  };

  return (
    <div className={styles.container}>
      <div className={styles.card}>
        <h1 className={styles.title}>Tell Us About Your Interests</h1>
        <p className={styles.subtitle}>
          Help us personalize your experience by selecting your preferences. You can skip this step and update it later.
        </p>

        <form onSubmit={handleSubmit} className={styles.form}>
          {/* Categories */}
          <div className={styles.section}>
            <label className={styles.label}>Categories of Interest</label>
            <div className={styles.checkboxGrid}>
              {CATEGORIES.map(category => (
                <label key={category} className={styles.checkboxLabel}>
                  <input
                    type="checkbox"
                    checked={preferences.selected_categories.includes(category)}
                    onChange={() => toggleSelection('selected_categories', category)}
                  />
                  <span>{category.charAt(0).toUpperCase() + category.slice(1).replace('_', ' ')}</span>
                </label>
              ))}
            </div>
          </div>

          {/* Fields */}
          <div className={styles.section}>
            <label className={styles.label}>Industries/Fields</label>
            <div className={styles.inputGroup}>
              <input
                type="text"
                value={newField}
                onChange={(e) => setNewField(e.target.value)}
                onKeyPress={(e) => {
                  if (e.key === 'Enter') {
                    e.preventDefault();
                    addCustomItem('selected_fields', newField, setNewField);
                  }
                }}
                placeholder="Add an industry or field (e.g., fintech, healthcare)"
                className={styles.input}
              />
              <button
                type="button"
                onClick={() => addCustomItem('selected_fields', newField, setNewField)}
                className={styles.addButton}
              >
                Add
              </button>
            </div>
            <div className={styles.tagList}>
              {preferences.selected_fields.map((field, idx) => (
                <span key={idx} className={styles.tag}>
                  {field}
                  <button
                    type="button"
                    onClick={() => removeItem('selected_fields', field)}
                    className={styles.removeTag}
                  >
                    ×
                  </button>
                </span>
              ))}
            </div>
          </div>

          {/* Tags */}
          <div className={styles.section}>
            <label className={styles.label}>Tags</label>
            <div className={styles.inputGroup}>
              <input
                type="text"
                value={newTag}
                onChange={(e) => setNewTag(e.target.value)}
                onKeyPress={(e) => {
                  if (e.key === 'Enter') {
                    e.preventDefault();
                    addCustomItem('selected_tags', newTag, setNewTag);
                  }
                }}
                placeholder="Add tags (e.g., AI, blockchain, mobile)"
                className={styles.input}
              />
              <button
                type="button"
                onClick={() => addCustomItem('selected_tags', newTag, setNewTag)}
                className={styles.addButton}
              >
                Add
              </button>
            </div>
            <div className={styles.tagList}>
              {preferences.selected_tags.map((tag, idx) => (
                <span key={idx} className={styles.tag}>
                  {tag}
                  <button
                    type="button"
                    onClick={() => removeItem('selected_tags', tag)}
                    className={styles.removeTag}
                  >
                    ×
                  </button>
                </span>
              ))}
            </div>
          </div>

          {/* Startup Stages */}
          <div className={styles.section}>
            <label className={styles.label}>Preferred Startup Stages</label>
            <div className={styles.inputGroup}>
              <input
                type="text"
                value={newStage}
                onChange={(e) => setNewStage(e.target.value)}
                onKeyPress={(e) => {
                  if (e.key === 'Enter') {
                    e.preventDefault();
                    addCustomItem('preferred_startup_stages', newStage, setNewStage);
                  }
                }}
                placeholder="Add stages (e.g., early, growth, mature)"
                className={styles.input}
              />
              <button
                type="button"
                onClick={() => addCustomItem('preferred_startup_stages', newStage, setNewStage)}
                className={styles.addButton}
              >
                Add
              </button>
            </div>
            <div className={styles.tagList}>
              {preferences.preferred_startup_stages.map((stage, idx) => (
                <span key={idx} className={styles.tag}>
                  {stage}
                  <button
                    type="button"
                    onClick={() => removeItem('preferred_startup_stages', stage)}
                    className={styles.removeTag}
                  >
                    ×
                  </button>
                </span>
              ))}
            </div>
          </div>

          {/* Engagement Types */}
          <div className={styles.section}>
            <label className={styles.label}>Preferred Engagement Types</label>
            <div className={styles.checkboxGrid}>
              {ENGAGEMENT_TYPES.map(type => (
                <label key={type} className={styles.checkboxLabel}>
                  <input
                    type="checkbox"
                    checked={preferences.preferred_engagement_types.includes(type)}
                    onChange={() => toggleSelection('preferred_engagement_types', type)}
                  />
                  <span>{type.charAt(0).toUpperCase() + type.slice(1).replace('-', ' ')}</span>
                </label>
              ))}
            </div>
          </div>

          {/* Preferred Skills */}
          <div className={styles.section}>
            <label className={styles.label}>Preferred Skills (for developers)</label>
            <div className={styles.inputGroup}>
              <input
                type="text"
                value={newSkill}
                onChange={(e) => setNewSkill(e.target.value)}
                onKeyPress={(e) => {
                  if (e.key === 'Enter') {
                    e.preventDefault();
                    addCustomItem('preferred_skills', newSkill, setNewSkill);
                  }
                }}
                placeholder="Add skills (e.g., Python, React, Machine Learning)"
                className={styles.input}
              />
              <button
                type="button"
                onClick={() => addCustomItem('preferred_skills', newSkill, setNewSkill)}
                className={styles.addButton}
              >
                Add
              </button>
            </div>
            <div className={styles.tagList}>
              {preferences.preferred_skills.map((skill, idx) => (
                <span key={idx} className={styles.tag}>
                  {skill}
                  <button
                    type="button"
                    onClick={() => removeItem('preferred_skills', skill)}
                    className={styles.removeTag}
                  >
                    ×
                  </button>
                </span>
              ))}
            </div>
          </div>

          {/* Buttons */}
          <div className={styles.buttonGroup}>
            <button
              type="button"
              onClick={handleSkip}
              className={styles.skipButton}
              disabled={loading}
            >
              Skip for Now
            </button>
            <button
              type="submit"
              className={styles.submitButton}
              disabled={loading}
            >
              {loading ? 'Saving...' : 'Save Preferences'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default OnboardingPreferences;

