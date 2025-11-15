import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Navbar } from '../../components/Navbar/Navbar';
import { Footer } from '../../components/Footer/Footer';
import { recommendationAPI } from '../../utils/apiServices';
import styles from './OnboardingPreferences.module.css';

const CATEGORIES = [
  'saas', 'ecommerce', 'agency', 'legal', 'marketplace',
  'media', 'platform', 'real_estate', 'robotics',
  'software', 'web3', 'crypto', 'other'
];

const ENGAGEMENT_TYPES = ['full-time', 'part-time', 'equity', 'paid'];
const INDUSTRIES = [
  'fintech', 'healthcare', 'edtech', 'climate', 'ai',
  'gaming', 'mobility', 'enterprise', 'consumer', 'creator-economy',
  'security', 'biotech', 'supply-chain', 'ar-vr', 'social-impact'
];
const TAG_SUGGESTIONS = [
  'ai', 'blockchain', 'mobile', 'web3', 'devtools', 'marketplace',
  'automation', 'e-commerce', 'subscription', 'market-network',
  'b2b', 'b2c', 'api-first', 'open-source', 'community-led'
];
const STARTUP_STAGES = [
  'idea', 'pre-seed', 'seed', 'post-seed', 'series-a',
  'series-b', 'growth', 'profitability', 'mature'
];
const SKILL_SUGGESTIONS = [
  'python', 'javascript', 'react', 'node.js', 'go',
  'java', 'swift', 'kotlin', 'ui/ux', 'product-management',
  'data-science', 'mlops', 'cloud-architecture', 'marketing', 'sales'
];

const inputPattern = /^[a-zA-Z0-9\s&+,\-()./]+$/;
const formatLabel = (value) =>
  value
    .replace(/[_-]/g, ' ')
    .replace(/\b\w/g, (char) => char.toUpperCase());

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
  const [errors, setErrors] = useState({});
  const [inputErrors, setInputErrors] = useState({});

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

  const clearFieldError = (field) => {
    setErrors((prev) => ({ ...prev, [field]: undefined }));
  };

  const toggleSelection = (field, value) => {
    setPreferences(prev => {
      const current = prev[field] || [];
      const updated = current.includes(value)
        ? current.filter(item => item !== value)
        : [...current, value];
      clearFieldError(field);
      return { ...prev, [field]: updated };
    });
  };

  const validateInputValue = (field, value) => {
    if (!value.trim()) {
      setInputErrors(prev => ({ ...prev, [field]: 'This field cannot be empty.' }));
      return false;
    }
    if (!inputPattern.test(value.trim())) {
      setInputErrors(prev => ({ ...prev, [field]: 'Only letters, numbers and basic symbols (.,-/&) are allowed.' }));
      return false;
    }
    setInputErrors(prev => ({ ...prev, [field]: undefined }));
    return true;
  };

  const addCustomItem = (field, value, setter) => {
    if (!validateInputValue(field, value)) return;
    const cleanedValue = value.trim();
    if (!preferences[field].includes(cleanedValue)) {
      setPreferences(prev => ({
        ...prev,
        [field]: [...prev[field], cleanedValue]
      }));
      clearFieldError(field);
      setter('');
    }
  };

  const removeItem = (field, value) => {
    setPreferences(prev => ({
      ...prev,
      [field]: prev[field].filter(item => item !== value)
    }));
  };

  const validatePreferences = () => {
    const newErrors = {};
    if (!preferences.selected_categories.length) newErrors.selected_categories = 'Select at least one category.';
    if (!preferences.selected_fields.length) newErrors.selected_fields = 'Add at least one industry or field.';
    if (!preferences.selected_tags.length) newErrors.selected_tags = 'Add at least one tag.';
    if (!preferences.preferred_startup_stages.length) newErrors.preferred_startup_stages = 'Add at least one preferred stage.';
    if (!preferences.preferred_engagement_types.length) newErrors.preferred_engagement_types = 'Select at least one engagement type.';
    if (!preferences.preferred_skills.length) newErrors.preferred_skills = 'Add at least one preferred skill.';
    return newErrors;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const validationErrors = validatePreferences();
    if (Object.keys(validationErrors).length > 0) {
      setErrors(validationErrors);
      return;
    }
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

  return (
    <>
      <Navbar />
      <div className={styles.container}>
        <div className={styles.card}>
        <h1 className={styles.title}>Tell Us About Your Interests</h1>
        <p className={styles.subtitle}>
          Help us personalize your experience by selecting your preferences. Fill every section so your matches stay accurate.
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
            {errors.selected_categories && (
              <p className={styles.errorText}>{errors.selected_categories}</p>
            )}
          </div>

          {/* Fields */}
          <div className={styles.section}>
            <label className={styles.label}>Industries/Fields</label>
            <p className={styles.sectionHint}>Choose all industries you’re excited about. We pre-filled the most common verticals.</p>
            <div className={styles.chipGrid}>
              {INDUSTRIES.map((industry) => {
                const selected = preferences.selected_fields.includes(industry);
                return (
                  <button
                    key={industry}
                    type="button"
                    className={`${styles.chipButton} ${selected ? styles.chipButtonSelected : ''}`}
                    onClick={() => toggleSelection('selected_fields', industry)}
                  >
                    {formatLabel(industry)}
                  </button>
                );
              })}
            </div>
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
                  {formatLabel(field)}
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
            {(errors.selected_fields || inputErrors.selected_fields) && (
              <p className={styles.errorText}>{errors.selected_fields || inputErrors.selected_fields}</p>
            )}
          </div>

          {/* Tags */}
          <div className={styles.section}>
            <label className={styles.label}>Tags</label>
            <p className={styles.sectionHint}>Tags help us match you with startup theses (tech stack, business model, GTM motion).</p>
            <div className={styles.chipGrid}>
              {TAG_SUGGESTIONS.map((tag) => {
                const selected = preferences.selected_tags.includes(tag);
                return (
                  <button
                    key={tag}
                    type="button"
                    className={`${styles.chipButton} ${selected ? styles.chipButtonSelected : ''}`}
                    onClick={() => toggleSelection('selected_tags', tag)}
                  >
                    {formatLabel(tag)}
                  </button>
                );
              })}
            </div>
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
                  {formatLabel(tag)}
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
            {(errors.selected_tags || inputErrors.selected_tags) && (
              <p className={styles.errorText}>{errors.selected_tags || inputErrors.selected_tags}</p>
            )}
          </div>

          {/* Startup Stages */}
          <div className={styles.section}>
            <label className={styles.label}>Preferred Startup Stages</label>
            <p className={styles.sectionHint}>Select the maturity levels where you can add the most value.</p>
            <div className={styles.chipGrid}>
              {STARTUP_STAGES.map((stage) => {
                const selected = preferences.preferred_startup_stages.includes(stage);
                return (
                  <button
                    key={stage}
                    type="button"
                    className={`${styles.chipButton} ${selected ? styles.chipButtonSelected : ''}`}
                    onClick={() => toggleSelection('preferred_startup_stages', stage)}
                  >
                    {formatLabel(stage)}
                  </button>
                );
              })}
            </div>
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
                  {formatLabel(stage)}
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
            {(errors.preferred_startup_stages || inputErrors.preferred_startup_stages) && (
              <p className={styles.errorText}>{errors.preferred_startup_stages || inputErrors.preferred_startup_stages}</p>
            )}
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
            {errors.preferred_engagement_types && (
              <p className={styles.errorText}>{errors.preferred_engagement_types}</p>
            )}
          </div>

          {/* Preferred Skills */}
          <div className={styles.section}>
            <label className={styles.label}>Preferred Skills (for developers)</label>
            <p className={styles.sectionHint}>Founders rely on this to tailor collaboration invites.</p>
            <div className={styles.chipGrid}>
              {SKILL_SUGGESTIONS.map((skill) => {
                const selected = preferences.preferred_skills.includes(skill);
                return (
                  <button
                    key={skill}
                    type="button"
                    className={`${styles.chipButton} ${selected ? styles.chipButtonSelected : ''}`}
                    onClick={() => toggleSelection('preferred_skills', skill)}
                  >
                    {formatLabel(skill)}
                  </button>
                );
              })}
            </div>
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
                  {formatLabel(skill)}
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
            {(errors.preferred_skills || inputErrors.preferred_skills) && (
              <p className={styles.errorText}>{errors.preferred_skills || inputErrors.preferred_skills}</p>
            )}
          </div>

          {/* Buttons */}
          <div className={styles.buttonGroup}>
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
      <Footer />
    </>
  );
};

export default OnboardingPreferences;

