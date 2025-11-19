import React, { useState, useEffect } from 'react';
import { useAuth } from '../../contexts/AuthContext';
import { Navbar } from '../../components/Navbar/Navbar';
import { Footer } from '../../components/Footer/Footer';
import { recommendationAPI, userAPI } from '../../utils/apiServices';
import { useRecommendationContext } from '../../hooks/useRecommendationContext';
import { toast } from 'react-toastify';
import styles from './PotentialInvestors.module.css';
import { Link } from 'react-router-dom';

const PotentialInvestors = () => {
  const { user, isEntrepreneur } = useAuth();
  const { storeSession } = useRecommendationContext();
  const [loading, setLoading] = useState(true);
  const [investors, setInvestors] = useState([]);
  const [myStartups, setMyStartups] = useState([]);
  const [selectedStartup, setSelectedStartup] = useState(null);
  const [currentPage, setCurrentPage] = useState(1);
  const [totalInvestors, setTotalInvestors] = useState(0);
  const investorsPerPage = 12;

  useEffect(() => {
    if (!isEntrepreneur()) {
      toast.error('Only entrepreneurs can view this page');
      return;
    }
    loadMyStartups();
  }, [isEntrepreneur]);

  useEffect(() => {
    if (selectedStartup) {
      loadRecommendedInvestors();
    }
  }, [selectedStartup, currentPage]);

  const loadMyStartups = async () => {
    try {
      const response = await userAPI.getUserStartups();
      const startups = response.data.results || response.data || [];
      
      if (startups.length === 0) {
        toast.info('You need to create a startup first to see investor recommendations');
        setLoading(false);
        return;
      }
      
      setMyStartups(startups);
      // Select first startup by default
      setSelectedStartup(startups[0]);
    } catch (error) {
      console.error('Failed to load startups:', error);
      toast.error('Failed to load your startups');
      setLoading(false);
    }
  };

  const loadRecommendedInvestors = async () => {
    if (!selectedStartup) return;
    
    try {
      setLoading(true);
      const response = await recommendationAPI.getPersonalizedInvestors(
        selectedStartup.id,
        { 
          limit: investorsPerPage,
          offset: (currentPage - 1) * investorsPerPage 
        }
      );
      
      console.log('Investor recommendations:', response.data);
      
      if (response.data.error) {
        toast.warning('Recommendation service temporarily unavailable');
        setInvestors([]);
      } else {
        const investorsList = response.data.investors || [];
        setInvestors(investorsList);
        setTotalInvestors(response.data.total || 0);
        
        // Store recommendation session for feedback tracking
        if (investorsList.length > 0 && selectedStartup) {
          const recommendations = investorsList.map((investor, index) => {
            const investorKey = investor?.id ? String(investor.id) : String(index);
            const score = response.data.scores?.[investorKey] ?? response.data.scores?.[investor?.id] ?? 0.0;
            return {
              user_id: investor.id,
              rank: index + 1,
              score: score,
              method: response.data.method_used || 'two_tower'
            };
          });
          
          storeSession({
            recommendations: recommendations,
            useCase: 'startup_investor',
            method: response.data.method_used || 'two_tower',
            modelVersion: response.data.model_version || 'two_tower_v1.0',
            startupId: selectedStartup.id
          }).catch(err => {
            console.error('[PotentialInvestors] Failed to store recommendation session:', err);
          });
        }
      }
    } catch (error) {
      console.error('Failed to load investor recommendations:', error);
      toast.error('Failed to load investor recommendations');
      setInvestors([]);
    } finally {
      setLoading(false);
    }
  };

  const handleStartupChange = (e) => {
    const startupId = e.target.value;
    const startup = myStartups.find(s => s.id === startupId);
    setSelectedStartup(startup);
    setCurrentPage(1); // Reset to first page
  };

  const totalPages = Math.ceil(totalInvestors / investorsPerPage);

  if (!isEntrepreneur()) {
    return null;
  }

  return (
    <>
      <Navbar />
      <div className={styles.container}>
        <div className={styles.header}>
          <h1>Potential Investors</h1>
          <p>Discover investors who might be interested in your startup</p>
        </div>

        {myStartups.length > 0 && (
          <div className={styles.filterSection}>
            <label htmlFor="startup-select" className={styles.filterLabel}>
              Select Startup:
            </label>
            <select
              id="startup-select"
              value={selectedStartup?.id || ''}
              onChange={handleStartupChange}
              className={styles.startupSelect}
            >
              {myStartups.map((startup) => (
                <option key={startup.id} value={startup.id}>
                  {startup.title}
                </option>
              ))}
            </select>
          </div>
        )}

        {loading ? (
          <div className={styles.loading}>
            <div className={styles.spinner}></div>
            <p>Loading recommended investors...</p>
          </div>
        ) : myStartups.length === 0 ? (
          <div className={styles.emptyState}>
            <h3>No Startups Found</h3>
            <p>Create a startup to see personalized investor recommendations</p>
            <Link to="/createstartup" className={styles.createButton}>
              Create Startup
            </Link>
          </div>
        ) : investors.length === 0 ? (
          <div className={styles.emptyState}>
            <h3>No Recommendations Available</h3>
            <p>We're working on finding the best investors for your startup. Check back soon!</p>
          </div>
        ) : (
          <>
            <div className={styles.investorsGrid}>
              {investors.map((investor) => (
                <div key={investor.id} className={styles.investorCard}>
                  <div className={styles.investorHeader}>
                    <div className={styles.investorAvatar}>
                      {investor.username?.charAt(0).toUpperCase() || 'I'}
                    </div>
                    <div className={styles.investorInfo}>
                      <h3>{investor.username}</h3>
                      <p className={styles.investorRole}>Investor</p>
                    </div>
                  </div>
                  
                  {investor.email && (
                    <div className={styles.investorDetail}>
                      <strong>Email:</strong> {investor.email}
                    </div>
                  )}
                  
                  {investor.bio && (
                    <div className={styles.investorBio}>
                      <strong>About:</strong>
                      <p>{investor.bio}</p>
                    </div>
                  )}
                  
                  <div className={styles.investorActions}>
                    <Link 
                      to={`/message?user=${investor.id}`} 
                      className={styles.contactButton}
                    >
                      Contact Investor
                    </Link>
                  </div>
                </div>
              ))}
            </div>

            {totalPages > 1 && (
              <div className={styles.pagination}>
                <button
                  onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
                  disabled={currentPage === 1}
                  className={styles.paginationButton}
                >
                  Previous
                </button>
                <span className={styles.pageInfo}>
                  Page {currentPage} of {totalPages}
                </span>
                <button
                  onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
                  disabled={currentPage === totalPages}
                  className={styles.paginationButton}
                >
                  Next
                </button>
              </div>
            )}
          </>
        )}
      </div>
      <Footer />
    </>
  );
};

export default PotentialInvestors;

