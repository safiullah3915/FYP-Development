import React, { useState, useEffect } from 'react';
import { Navbar } from '../../components/Navbar/Navbar';
import { Footer } from '../../components/Footer/Footer';
import { useAuth } from '../../contexts/AuthContext';
import { useNavigate } from 'react-router-dom';
import { toast } from 'react-toastify';
import apiClient from '../../utils/axiosConfig';
import styles from './PitchIdea.module.css';

const PitchIdea = () => {
  const { user, isEntrepreneur } = useAuth();
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [investors, setInvestors] = useState([]);
  const [selectedInvestors, setSelectedInvestors] = useState([]);
  const [pitchData, setPitchData] = useState({
    title: '',
    executive_summary: '',
    problem_statement: '',
    solution: '',
    market_size: '',
    business_model: '',
    funding_needed: '',
    use_of_funds: '',
    pitch_deck_url: '',
    video_pitch_url: '',
    contact_email: user?.email || '',
    contact_phone: ''
  });

  useEffect(() => {
    if (!isEntrepreneur()) {
      toast.error('Only entrepreneurs can pitch business ideas');
      navigate('/dashboard');
      return;
    }
    loadInvestors();
  }, [isEntrepreneur, navigate]);

  const loadInvestors = async () => {
    try {
      // Get list of investors (using online users endpoint as proxy)
      const response = await apiClient.get('/api/messages/users/online');
      const investorUsers = response.data.filter(user => user.role === 'investor');
      setInvestors(investorUsers);
    } catch (error) {
      console.error('Failed to load investors:', error);
      // Don't show error to user, investors list is optional
    }
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setPitchData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleInvestorToggle = (investorId) => {
    setSelectedInvestors(prev => {
      if (prev.includes(investorId)) {
        return prev.filter(id => id !== investorId);
      } else {
        return [...prev, investorId];
      }
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!pitchData.title.trim() || !pitchData.executive_summary.trim() || !pitchData.problem_statement.trim()) {
      toast.error('Please fill in all required fields');
      return;
    }

    if (selectedInvestors.length === 0) {
      toast.error('Please select at least one investor to pitch to');
      return;
    }

    setLoading(true);

    try {
      // Create a startup entry for the pitch
      const startupPayload = {
        title: pitchData.title,
        description: pitchData.executive_summary,
        field: pitchData.market_size,
        type: 'marketplace',
        category: 'other',
        asking_price: pitchData.funding_needed,
        website_url: pitchData.pitch_deck_url || null,
        // Store additional pitch data in unused fields temporarily
        ttm_revenue: pitchData.problem_statement,
        ttm_profit: pitchData.solution,
        last_month_revenue: pitchData.business_model,
        last_month_profit: pitchData.use_of_funds
      };

      const startupResponse = await apiClient.post('/api/startups', startupPayload);
      const startupId = startupResponse.data.id;

      // Send notifications to selected investors
      const notificationPromises = selectedInvestors.map(investorId => 
        apiClient.post('/api/notifications', {
          user_id: investorId,
          type: 'pitch',
          title: `New Business Pitch: ${pitchData.title}`,
          message: `${user.username} has pitched their business idea "${pitchData.title}" to you. Check it out in the marketplace!`,
          data: {
            startup_id: startupId,
            entrepreneur_id: user.id,
            contact_email: pitchData.contact_email,
            contact_phone: pitchData.contact_phone,
            pitch_deck_url: pitchData.pitch_deck_url,
            video_pitch_url: pitchData.video_pitch_url
          }
        }).catch(err => {
          console.warn('Failed to send notification to investor:', investorId, err);
          return null; // Don't fail entire pitch if one notification fails
        })
      );

      await Promise.allSettled(notificationPromises);

      toast.success('Pitch submitted successfully to selected investors!');
      navigate('/dashboard');
    } catch (error) {
      console.error('Failed to submit pitch:', error);
      toast.error('Failed to submit pitch. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <Navbar />
      <div className={styles.container}>
        <div className={styles.header}>
          <h1>Pitch Your Business Idea</h1>
          <p>Present your startup concept to potential investors</p>
        </div>

        <form onSubmit={handleSubmit} className={styles.form}>
          <div className={styles.section}>
            <h2>Basic Information</h2>
            
            <div className={styles.formGroup}>
              <label>Business Idea Title *</label>
              <input
                type="text"
                name="title"
                value={pitchData.title}
                onChange={handleInputChange}
                placeholder="Enter your business idea title"
                required
              />
            </div>

            <div className={styles.formGroup}>
              <label>Executive Summary *</label>
              <textarea
                name="executive_summary"
                value={pitchData.executive_summary}
                onChange={handleInputChange}
                placeholder="Provide a brief overview of your business idea (elevator pitch)"
                rows="4"
                required
              />
            </div>
          </div>

          <div className={styles.section}>
            <h2>Business Details</h2>
            
            <div className={styles.row}>
              <div className={styles.formGroup}>
                <label>Problem Statement *</label>
                <textarea
                  name="problem_statement"
                  value={pitchData.problem_statement}
                  onChange={handleInputChange}
                  placeholder="What problem does your business solve?"
                  rows="3"
                  required
                />
              </div>
              <div className={styles.formGroup}>
                <label>Solution *</label>
                <textarea
                  name="solution"
                  value={pitchData.solution}
                  onChange={handleInputChange}
                  placeholder="How does your business solve the problem?"
                  rows="3"
                  required
                />
              </div>
            </div>

            <div className={styles.row}>
              <div className={styles.formGroup}>
                <label>Market Size & Target</label>
                <input
                  type="text"
                  name="market_size"
                  value={pitchData.market_size}
                  onChange={handleInputChange}
                  placeholder="e.g., $1B market, Tech professionals, SMBs"
                />
              </div>
              <div className={styles.formGroup}>
                <label>Business Model</label>
                <input
                  type="text"
                  name="business_model"
                  value={pitchData.business_model}
                  onChange={handleInputChange}
                  placeholder="e.g., SaaS subscription, Marketplace commission"
                />
              </div>
            </div>
          </div>

          <div className={styles.section}>
            <h2>Funding Information</h2>
            
            <div className={styles.row}>
              <div className={styles.formGroup}>
                <label>Funding Needed</label>
                <input
                  type="text"
                  name="funding_needed"
                  value={pitchData.funding_needed}
                  onChange={handleInputChange}
                  placeholder="e.g., $100,000, $500K, $1M"
                />
              </div>
              <div className={styles.formGroup}>
                <label>Use of Funds</label>
                <input
                  type="text"
                  name="use_of_funds"
                  value={pitchData.use_of_funds}
                  onChange={handleInputChange}
                  placeholder="e.g., Product development, Marketing, Team hiring"
                />
              </div>
            </div>
          </div>

          <div className={styles.section}>
            <h2>Supporting Materials</h2>
            
            <div className={styles.row}>
              <div className={styles.formGroup}>
                <label>Pitch Deck URL</label>
                <input
                  type="url"
                  name="pitch_deck_url"
                  value={pitchData.pitch_deck_url}
                  onChange={handleInputChange}
                  placeholder="https://your-pitch-deck.pdf"
                />
              </div>
              <div className={styles.formGroup}>
                <label>Video Pitch URL</label>
                <input
                  type="url"
                  name="video_pitch_url"
                  value={pitchData.video_pitch_url}
                  onChange={handleInputChange}
                  placeholder="https://youtube.com/your-pitch-video"
                />
              </div>
            </div>

            <div className={styles.row}>
              <div className={styles.formGroup}>
                <label>Contact Email *</label>
                <input
                  type="email"
                  name="contact_email"
                  value={pitchData.contact_email}
                  onChange={handleInputChange}
                  placeholder="your.email@example.com"
                  required
                />
              </div>
              <div className={styles.formGroup}>
                <label>Contact Phone</label>
                <input
                  type="tel"
                  name="contact_phone"
                  value={pitchData.contact_phone}
                  onChange={handleInputChange}
                  placeholder="+1 (555) 123-4567"
                />
              </div>
            </div>
          </div>

          <div className={styles.section}>
            <h2>Select Investors to Pitch To *</h2>
            <p>Choose investors who might be interested in your business idea</p>
            
            {investors.length === 0 ? (
              <div className={styles.noInvestors}>
                <p>No investors found. Your pitch will be submitted to the general marketplace.</p>
              </div>
            ) : (
              <div className={styles.investorsList}>
                {investors.map(investor => (
                  <div key={investor.id} className={styles.investorCard}>
                    <label className={styles.investorLabel}>
                      <input
                        type="checkbox"
                        checked={selectedInvestors.includes(investor.id)}
                        onChange={() => handleInvestorToggle(investor.id)}
                      />
                      <div className={styles.investorInfo}>
                        <strong>{investor.username}</strong>
                        <span>{investor.email}</span>
                      </div>
                    </label>
                  </div>
                ))}
              </div>
            )}
          </div>

          <div className={styles.submitSection}>
            <button
              type="button"
              onClick={() => navigate('/dashboard')}
              className={styles.cancelButton}
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={loading || (investors.length > 0 && selectedInvestors.length === 0)}
              className={styles.submitButton}
            >
              {loading ? 'Submitting Pitch...' : 'Submit Pitch to Investors'}
            </button>
          </div>
        </form>
      </div>
      <Footer />
    </>
  );
};

export default PitchIdea;