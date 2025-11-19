import React, { useState, useEffect } from "react";
import { Navbar } from '../../components/Navbar/Navbar';
import { Footer } from '../../components/Footer/Footer';
import CollaborationCard from "../../components/CollaborationCard/CollaborationCard";
import JobCard from "../../components/JobCard/JobCard";
import styles from "./Collaboration.module.css";
import { positionAPI, userAPI, recommendationAPI } from '../../utils/apiServices';
import { useAuth } from '../../contexts/AuthContext';
import { useRecommendationContext } from '../../hooks/useRecommendationContext';
import { toast } from 'react-toastify';

const Collaboration = () => {
  const [positions, setPositions] = useState([]);
  const [currentPage, setCurrentPage] = useState(1);
  const [loading, setLoading] = useState(true);
  const [showFilters, setShowFilters] = useState(false);
  const [filters, setFilters] = useState({
    category: '',
    field: '',
    phase: '',
    team_size: '',
    query: ''
  });
  const [appliedPositionIds, setAppliedPositionIds] = useState(new Set());
  const [appliedStartupIds, setAppliedStartupIds] = useState(new Set());
  const { isStudent, isInvestor, isEntrepreneur, user } = useAuth();
  const { storeSession, getRecommendationContext } = useRecommendationContext();
  const positionsPerPage = 9;
  
  // New states for recommendation mode
  const [viewMode, setViewMode] = useState('recommended'); // 'recommended' | 'all'
  const [recommendedStartups, setRecommendedStartups] = useState([]);
  const [recommendationScores, setRecommendationScores] = useState({});
  const [recommendationReasons, setRecommendationReasons] = useState({});
  const [recommendedLoading, setRecommendedLoading] = useState(false);
  const [recommendedCurrentPage, setRecommendedCurrentPage] = useState(1);
  const [totalRecommended, setTotalRecommended] = useState(0);
  const startupsPerPage = 20;

  // Categories from backend CATEGORY_CHOICES
  const categories = [
    { value: '', label: 'All Categories' },
    { value: 'saas', label: 'SaaS' },
    { value: 'ecommerce', label: 'E-commerce' },
    { value: 'agency', label: 'Agency' },
    { value: 'legal', label: 'Legal' },
    { value: 'marketplace', label: 'Marketplace' },
    { value: 'media', label: 'Media' },
    { value: 'platform', label: 'Platform' },
    { value: 'real_estate', label: 'Real Estate' },
    { value: 'robotics', label: 'Robotics' },
    { value: 'software', label: 'Software' },
    { value: 'web3', label: 'Web3' },
    { value: 'crypto', label: 'Crypto' },
    { value: 'other', label: 'Other' }
  ];

  // Phases from startup creation form stages
  const phases = [
    { value: '', label: 'All Phases' },
    { value: 'Idea Stage', label: 'Idea Stage' },
    { value: 'Building MVP', label: 'Building MVP' },
    { value: 'MVP Stage', label: 'MVP Stage' },
    { value: 'Product Market Fit', label: 'Product Market Fit' },
    { value: 'Fund raising', label: 'Fund raising' },
    { value: 'Growth', label: 'Growth' }
  ];

  useEffect(() => {
    if (viewMode === 'all') {
      loadPositions();
    }
  }, [filters, viewMode]);

  useEffect(() => {
    if (viewMode === 'recommended') {
      loadRecommendedStartups();
    }
  }, [viewMode, filters.query, recommendedCurrentPage]);

  useEffect(() => {
    setRecommendedCurrentPage(1);
  }, [filters.query]);

  useEffect(() => {
    const total = Math.max(1, Math.ceil(positions.length / positionsPerPage));
    if (currentPage > total) {
      setCurrentPage(total);
    }
  }, [positions, currentPage, positionsPerPage]);

  useEffect(() => {
    // Check applied positions when component mounts
    checkAppliedPositions();
  }, []);

  const loadPositions = async () => {
    try {
      setLoading(true);
      const params = {};
      
      // Add non-empty filter values to params
      Object.entries(filters).forEach(([key, value]) => {
        if (value && value.trim()) {
          params[key] = value;
        }
      });

      const response = await positionAPI.getAllPositions(params);
      
      let allPositions = response.data.results || [];
      
      // Filter out positions from startups owned by the current entrepreneur
      if (isEntrepreneur() && user?.id) {
        allPositions = allPositions.filter(position => {
          return position.startup?.owner?.id !== user.id;
        });
      }
      
      setPositions(allPositions);
      setCurrentPage(1);
    } catch (error) {
      console.error('Failed to load positions:', error);
      setPositions([]);
    } finally {
      setLoading(false);
    }
  };

  const loadRecommendedStartups = async () => {
    try {
      setRecommendedLoading(true);
      const params = {
        limit: startupsPerPage,
        offset: (recommendedCurrentPage - 1) * startupsPerPage,
        require_open_positions: true
      };
      
      // Add search query if present
      if (filters.query && filters.query.trim()) {
        params.query = filters.query;
      }
      
      const response = await recommendationAPI.getPersonalizedCollaborationStartups(params);
      
      console.log('[Collaboration] Recommended startups response:', response.data);
      
      if (response.data.error) {
        toast.warning('Recommendation service temporarily unavailable. Switching to all opportunities.');
        setViewMode('all');
        return;
      }
      
      // Optional UI fallback: if we received IDs but zero hydrated startups, retry once without open-position requirement
      if ((!response.data.startups || response.data.startups.length === 0) && Array.isArray(response.data.startup_ids) && response.data.startup_ids.length > 0 && params.require_open_positions) {
        console.log('[Collaboration] Empty hydrated startups with non-empty startup_ids, retrying without require_open_positions');
        const fallbackResp = await recommendationAPI.getPersonalizedCollaborationStartups({
          ...params,
          require_open_positions: false
        });
        if (!fallbackResp.data.error) {
          const fbStartups = fallbackResp.data.startups || [];
          setRecommendedStartups(fbStartups);
          setTotalRecommended(
            typeof fallbackResp.data.total === 'number'
              ? fallbackResp.data.total
              : fbStartups.length
          );
          setRecommendationScores(fallbackResp.data.scores || {});
          setRecommendationReasons(fallbackResp.data.match_reasons || {});
          
          // Store recommendation session for fallback response too
          if (fbStartups.length > 0) {
            const recommendations = fbStartups.map((startup, index) => {
              const startupKey = startup?.id ? String(startup.id) : String(index);
              const score = fallbackResp.data.scores?.[startupKey] ?? fallbackResp.data.scores?.[startup?.id] ?? 0.0;
              return {
                startup_id: startup.id,
                rank: index + 1,
                score: score,
                method: fallbackResp.data.method_used || 'two_tower'
              };
            });
            
            storeSession({
              recommendations: recommendations,
              useCase: 'developer_startup',
              method: fallbackResp.data.method_used || 'two_tower',
              modelVersion: fallbackResp.data.model_version || 'two_tower_v1.0'
            }).catch(err => {
              console.error('[Collaboration] Failed to store recommendation session (fallback):', err);
            });
          }
          return;
        }
      }
      
      const startupsResponse = response.data.startups || [];
      setRecommendedStartups(startupsResponse);
      setTotalRecommended(
        typeof response.data.total === 'number'
          ? response.data.total
          : startupsResponse.length
      );
      setRecommendationScores(response.data.scores || {});
      setRecommendationReasons(response.data.match_reasons || {});
      
      // Store recommendation session for feedback tracking
      if (startupsResponse.length > 0) {
        const recommendations = startupsResponse.map((startup, index) => {
          const startupKey = startup?.id ? String(startup.id) : String(index);
          const score = response.data.scores?.[startupKey] ?? response.data.scores?.[startup?.id] ?? 0.0;
          return {
            startup_id: startup.id,
            rank: index + 1,
            score: score,
            method: response.data.method_used || 'two_tower'
          };
        });
        
        storeSession({
          recommendations: recommendations,
          useCase: 'developer_startup',
          method: response.data.method_used || 'two_tower',
          modelVersion: response.data.model_version || 'two_tower_v1.0'
        }).catch(err => {
          console.error('[Collaboration] Failed to store recommendation session:', err);
          // Don't block UI if session storage fails
        });
      }
    } catch (error) {
      console.error('Failed to load recommended startups:', error);
      toast.error('Failed to load personalized recommendations');
      setRecommendedStartups([]);
    } finally {
      setRecommendedLoading(false);
    }
  };

  const handleFilterChange = (filterType, value) => {
    setFilters(prev => ({
      ...prev,
      [filterType]: value
    }));
  };

  const toggleFilters = () => {
    setShowFilters((prev) => !prev);
  };

  const checkAppliedPositions = async () => {
    // Only check for students and entrepreneurs (not investors)
    if (!user || isInvestor()) return;
    
    try {
      const response = await userAPI.getUserApplications();
      console.log('[Collaboration] Checking applied positions');
      console.log('[Collaboration] Applications response:', response.data);
      
      // Handle both paginated and non-paginated responses
      const applications = response.data.results || response.data;
      
      if (Array.isArray(applications)) {
        // Extract position IDs from all applications
        const appliedIds = new Set(
          applications
            .filter(app => app.position && app.position.id)
            .map(app => app.position.id)
        );
        
        // Extract startup IDs (to check if user has applied to any position in a startup)
        const appliedStartupIdsSet = new Set(
          applications
            .filter(app => app.startup && app.startup.id)
            .map(app => app.startup.id)
        );
        
        console.log('[Collaboration] Applied position IDs:', Array.from(appliedIds));
        console.log('[Collaboration] Applied startup IDs:', Array.from(appliedStartupIdsSet));
        setAppliedPositionIds(appliedIds);
        setAppliedStartupIds(appliedStartupIdsSet);
      }
    } catch (error) {
      console.error('Failed to check applied positions:', error);
      // Don't show error to user for this non-critical operation
    }
  };

  const startIndex = (currentPage - 1) * positionsPerPage;
  const paginatedPositions = positions.slice(startIndex, startIndex + positionsPerPage);
  const totalPages = Math.ceil(positions.length / positionsPerPage);

  const getVisiblePages = () => {
    if (totalPages <= 3) {
      return Array.from({ length: totalPages }, (_, index) => index + 1);
    }

    if (currentPage <= 2) {
      return [1, 2, 3];
    }

    if (currentPage >= totalPages - 1) {
      return [totalPages - 2, totalPages - 1, totalPages];
    }

    return [currentPage - 1, currentPage, currentPage + 1];
  };

  const handlePageChange = (page) => {
    if (page === currentPage || page < 1 || page > totalPages) return;
    setCurrentPage(page);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  return (
    <>
    <div className={styles.main}>

    
      <Navbar />
      <div className={styles.marketplace}>
        <div className={styles.header}>
          <h2>{viewMode === 'recommended' ? 'Recommended jobs for you' : 'Search all jobs'}</h2>

          {/* Tab System */}
          <div className={styles.tabContainer}>
            <button
              className={`${styles.tab} ${viewMode === 'recommended' ? styles.activeTab : ''}`}
              onClick={() => {
                setViewMode('recommended');
                setRecommendedCurrentPage(1);
              }}
            >
              Recommended Jobs
            </button>
            <button
              className={`${styles.tab} ${viewMode === 'all' ? styles.activeTab : ''}`}
              onClick={() => {
                setViewMode('all');
                setCurrentPage(1);
              }}
            >
              Search All Jobs
            </button>
          </div>

          {/* Search Bar */}
          <div className={styles.searchSection}>
            <input
              type="text"
              placeholder={viewMode === 'recommended' ? "Search recommended jobs..." : "Search jobs by title, description, or company..."}
              value={filters.query}
              onChange={(e) => handleFilterChange('query', e.target.value)}
              className={styles.searchInput}
            />
          </div>

          {/* Mobile Filter Toggle Button - Only show in 'all' mode */}
          {viewMode === 'all' && (
            <button className={styles.filterToggle} onClick={toggleFilters}>
              {showFilters ? "Hide Filters" : "Show Filters"}
            </button>
          )}

          {/* Filters Section - Only show in 'all' mode */}
          {viewMode === 'all' && (
            <div
              className={`${styles.filters} ${
                showFilters ? styles.showFilters : ""
              }`}
            >
            <select 
              value={filters.category} 
              onChange={(e) => handleFilterChange('category', e.target.value)}
            >
              {categories.map((cat) => (
                <option key={cat.value} value={cat.value}>
                  {cat.label}
                </option>
              ))}
            </select>
            
            {/*<input
              type="text"
              placeholder="Field/Industry"
              value={filters.field}
              onChange={(e) => handleFilterChange('field', e.target.value)}
              className={styles.filterInput}
            />*/}
            
            <select 
              value={filters.phase} 
              onChange={(e) => handleFilterChange('phase', e.target.value)}
            >
              {phases.map((phase) => (
                <option key={phase.value} value={phase.value}>
                  {phase.label}
                </option>
              ))}
            </select>
            
            <select 
              value={filters.team_size} 
              onChange={(e) => handleFilterChange('team_size', e.target.value)}
            >
              <option value="">All Team Sizes</option>
              <option value="1">Just me</option>
              <option value="2-5">2-5 people</option>
              <option value="6-10">6-10 people</option>
              <option value="11-25">11-25 people</option>
              <option value="25+">25+ people</option>
            </select>
          </div>
          )}
        </div>

        {/* Cards Grid - Conditional rendering based on viewMode */}
        {viewMode === 'recommended' ? (
          // Recommended Startups View
          <div className={styles.cardsGrid}>
            {recommendedLoading ? (
              <div className={styles.loading}>Loading personalized recommendations...</div>
            ) : recommendedStartups.length > 0 ? (
              recommendedStartups.map((startup, index) => {
                const startupKey = startup?.id ? String(startup.id) : String(index);
                const score = recommendationScores[startupKey] ?? recommendationScores[startup?.id];
                const reasons = recommendationReasons[startupKey] ?? recommendationReasons[startup?.id];
                
                // Get recommendation context for this startup
                const recommendationContext = getRecommendationContext(startup.id);
                
                return (
                  <CollaborationCard
                    key={startup.id || index}
                    startup={startup}
                    score={score}
                    matchReasons={reasons}
                    appliedPositionIds={appliedPositionIds}
                    appliedStartupIds={appliedStartupIds}
                    recommendationContext={recommendationContext}
                  />
                );
              })
            ) : (
              <div className={styles.noResults}>
                <h3>No recommendations available</h3>
                <p>We're building your personalized recommendations. Try the "Search All Jobs" tab to browse every open role.</p>
              </div>
            )}
          </div>
        ) : (
          // All Positions View
          <div className={styles.cardsGrid}>
            {loading ? (
              <div className={styles.loading}>Loading job opportunities...</div>
            ) : positions.length > 0 ? (
              paginatedPositions.map((position, index) => {
                const hasAppliedToPosition = appliedPositionIds.has(position.id);
                const hasAppliedToStartup = position.startup?.id && appliedStartupIds.has(position.startup.id);
                return (
                  <JobCard 
                    key={position.id || index} 
                    {...position} 
                    isApplied={hasAppliedToPosition || hasAppliedToStartup}
                    hasAppliedToSpecificPosition={hasAppliedToPosition}
                    hasAppliedToStartup={hasAppliedToStartup && !hasAppliedToPosition}
                  />
                );
              })
            ) : (
              <div className={styles.noResults}>
                <h3>No job opportunities found</h3>
                <p>Try adjusting your filters or search terms, or check back later for new positions.</p>
              </div>
            )}
          </div>
        )}
        
        {/* Pagination for Recommended Mode */}
        {viewMode === 'recommended' && totalRecommended > startupsPerPage && (
          <div className={styles.pagination}>
            <button
              type="button"
              onClick={() => {
                setRecommendedCurrentPage(p => Math.max(1, p - 1));
                window.scrollTo({ top: 0, behavior: 'smooth' });
              }}
              disabled={recommendedCurrentPage === 1}
            >
              &larr;
            </button>
            <span>Page {recommendedCurrentPage} of {Math.ceil(totalRecommended / startupsPerPage)}</span>
            <button
              type="button"
              onClick={() => {
                setRecommendedCurrentPage(p => Math.min(Math.ceil(totalRecommended / startupsPerPage), p + 1));
                window.scrollTo({ top: 0, behavior: 'smooth' });
              }}
              disabled={recommendedCurrentPage >= Math.ceil(totalRecommended / startupsPerPage)}
            >
              &rarr;
            </button>
          </div>
        )}
        
        {/* Pagination for All Mode */}
        {viewMode === 'all' && positions.length > 0 && totalPages > 1 && (
          <div className={styles.pagination}>
            <button
              type="button"
              onClick={() => handlePageChange(currentPage - 1)}
              disabled={currentPage === 1}
            >
              &larr;
            </button>
            {totalPages > 3 && currentPage > 3 && <span>...</span>}
            {getVisiblePages().map((page) => (
              <button
                type="button"
                key={page}
                onClick={() => handlePageChange(page)}
                disabled={page === currentPage}
              >
                {page}
              </button>
            ))}
            {totalPages > 3 && currentPage < totalPages - 2 && <span>...</span>}
            <button
              type="button"
              onClick={() => handlePageChange(currentPage + 1)}
              disabled={currentPage === totalPages}
            >
              &rarr;
            </button>
          </div>
        )}
      </div>
      </div>
      <Footer />
    </>
  );
};

export { Collaboration };
