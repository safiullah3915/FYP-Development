import React, { useState, useEffect } from 'react';
import { useAuth } from '../../contexts/AuthContext';
import { Navbar } from '../../components/Navbar/Navbar';
import { Footer } from '../../components/Footer/Footer';
import { recommendationAPI, userAPI } from '../../utils/apiServices';
import { useRecommendationContext } from '../../hooks/useRecommendationContext';
import { toast } from 'react-toastify';
import styles from './MatchedDevelopers.module.css';
import { Link } from 'react-router-dom';

const MatchedDevelopers = () => {
  const { user, isEntrepreneur } = useAuth();
  const { storeSession } = useRecommendationContext();
  const [loading, setLoading] = useState(true);
  const [developers, setDevelopers] = useState([]);
  const [myStartups, setMyStartups] = useState([]);
  const [selectedStartup, setSelectedStartup] = useState(null);
  const [currentPage, setCurrentPage] = useState(1);
  const [totalDevelopers, setTotalDevelopers] = useState(0);
  const developersPerPage = 12;

  useEffect(() => {
    if (!isEntrepreneur()) {
      toast.error('Only entrepreneurs can view this page');
      return;
    }
    loadMyStartups();
  }, [isEntrepreneur]);

  useEffect(() => {
    if (selectedStartup) {
      loadRecommendedDevelopers();
    }
  }, [selectedStartup, currentPage]);

  const loadMyStartups = async () => {
    try {
      const response = await userAPI.getUserStartups();
      const startups = response.data.results || response.data || [];
      
      // Filter only collaboration startups
      const collaborationStartups = startups.filter(s => s.type === 'collaboration');
      
      if (collaborationStartups.length === 0) {
        toast.info('You need to create a collaboration startup first to see developer recommendations');
        setLoading(false);
        return;
      }
      
      setMyStartups(collaborationStartups);
      // Select first startup by default
      setSelectedStartup(collaborationStartups[0]);
    } catch (error) {
      console.error('Failed to load startups:', error);
      toast.error('Failed to load your startups');
      setLoading(false);
    }
  };

  const loadRecommendedDevelopers = async () => {
    if (!selectedStartup) return;
    
    try {
      setLoading(true);
      const response = await recommendationAPI.getPersonalizedDevelopers(
        selectedStartup.id,
        { 
          limit: developersPerPage,
          offset: (currentPage - 1) * developersPerPage 
        }
      );
      
      console.log('Developer recommendations:', response.data);
      
      if (response.data.error) {
        toast.warning('Recommendation service temporarily unavailable');
        setDevelopers([]);
      } else {
        const developersList = response.data.developers || [];
        setDevelopers(developersList);
        setTotalDevelopers(response.data.total || 0);
        
        // Store recommendation session for feedback tracking
        if (developersList.length > 0 && selectedStartup) {
          const recommendations = developersList.map((developer, index) => {
            const developerKey = developer?.id ? String(developer.id) : String(index);
            const score = response.data.scores?.[developerKey] ?? response.data.scores?.[developer?.id] ?? 0.0;
            return {
              user_id: developer.id,
              rank: index + 1,
              score: score,
              method: response.data.method_used || 'two_tower'
            };
          });
          
          storeSession({
            recommendations: recommendations,
            useCase: 'startup_developer',
            method: response.data.method_used || 'two_tower',
            modelVersion: response.data.model_version || 'two_tower_v1.0',
            startupId: selectedStartup.id
          }).catch(err => {
            console.error('[MatchedDevelopers] Failed to store recommendation session:', err);
          });
        }
      }
    } catch (error) {
      console.error('Failed to load developer recommendations:', error);
      toast.error('Failed to load developer recommendations');
      setDevelopers([]);
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

  const totalPages = Math.ceil(totalDevelopers / developersPerPage);

  if (!isEntrepreneur()) {
    return null;
  }

  return (
    <>
      <Navbar />
      <div className={styles.container}>
        <div className={styles.header}>
          <h1>Matched Developers</h1>
          <p>Find developers that match your startup's open positions</p>
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
            <p>Loading matched developers...</p>
          </div>
        ) : myStartups.length === 0 ? (
          <div className={styles.emptyState}>
            <h3>No Collaboration Startups Found</h3>
            <p>Create a collaboration startup with open positions to see personalized developer recommendations</p>
            <Link to="/createstartup" className={styles.createButton}>
              Create Startup
            </Link>
          </div>
        ) : developers.length === 0 ? (
          <div className={styles.emptyState}>
            <h3>No Recommendations Available</h3>
            <p>We're working on finding the best developers for your startup positions. Check back soon!</p>
            <Link to={`/startups/${selectedStartup?.id}/positions`} className={styles.manageButton}>
              Manage Positions
            </Link>
          </div>
        ) : (
          <>
            <div className={styles.developersGrid}>
              {developers.map((developer) => (
                <div key={developer.id} className={styles.developerCard}>
                  <div className={styles.developerHeader}>
                    <div className={styles.developerAvatar}>
                      {developer.username?.charAt(0).toUpperCase() || 'D'}
                    </div>
                    <div className={styles.developerInfo}>
                      <h3>{developer.username}</h3>
                      <p className={styles.developerRole}>
                        {developer.role === 'student' ? 'Student' : 'Developer'}
                      </p>
                    </div>
                  </div>
                  
                  {developer.email && (
                    <div className={styles.developerDetail}>
                      <strong>Email:</strong> {developer.email}
                    </div>
                  )}
                  
                  {developer.skills && developer.skills.length > 0 && (
                    <div className={styles.skillsSection}>
                      <strong>Skills:</strong>
                      <div className={styles.skillsTags}>
                        {developer.skills.map((skill, idx) => (
                          <span key={idx} className={styles.skillTag}>
                            {skill}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                  
                  {developer.bio && (
                    <div className={styles.developerBio}>
                      <strong>About:</strong>
                      <p>{developer.bio}</p>
                    </div>
                  )}
                  
                  <div className={styles.developerActions}>
                    <Link 
                      to={`/message?user=${developer.id}`} 
                      className={styles.contactButton}
                    >
                      Contact Developer
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

export default MatchedDevelopers;

