import React, { useState, useEffect } from "react";
import { Navbar } from '../../components/Navbar/Navbar';
import { Footer } from '../../components/Footer/Footer';
import MarketPlaceCard from "../../components/MarketplaceCard/MarketPlaceCard";
import styles from "./MarketPlace.module.css";
import { useAuth } from '../../contexts/AuthContext';
import apiClient from '../../utils/axiosConfig';
import { recommendationAPI, marketplaceAPI } from '../../utils/apiServices';
import { toast } from 'react-toastify';

const Marketplace = () => {
  const [startups, setStartups] = useState([]);
  const [currentPage, setCurrentPage] = useState(1);
  const [loading, setLoading] = useState(true);
  const startupsPerPage = 9;
  const [filters, setFilters] = useState({
    sortBy: 'date',
    order: 'desc',
    type: '',
    minRevenue: '',
    maxRevenue: '',
    category: ''
  });
  const { isInvestor, isStudent, user } = useAuth();
  
  // New states for recommendation mode (only for investors)
  // Initialize to 'recommended' for investors, 'all' for others
  const [viewMode, setViewMode] = useState(() => {
    // This will be evaluated once on mount
    return isInvestor() ? 'recommended' : 'all';
  });
  const [recommendedStartups, setRecommendedStartups] = useState([]);
  const [recommendationScores, setRecommendationScores] = useState({});
  const [recommendationReasons, setRecommendationReasons] = useState({});
  const [recommendedLoading, setRecommendedLoading] = useState(false);
  const [recommendedCurrentPage, setRecommendedCurrentPage] = useState(1);
  const [totalRecommended, setTotalRecommended] = useState(0);
  const recommendedStartupsPerPage = 20;

  useEffect(() => {
    if (viewMode === 'all') {
      loadStartups();
    }
  }, [filters, viewMode]);

  useEffect(() => {
    if (viewMode === 'recommended' && isInvestor()) {
      loadRecommendedStartups();
    }
  }, [viewMode, filters.category, recommendedCurrentPage]);

  useEffect(() => {
    if (viewMode === 'recommended') {
      setRecommendedCurrentPage(1);
    }
  }, [filters.category]);

  const loadStartups = async () => {
    try {
      setLoading(true);
      const params = {};
      
      if (filters.sortBy) params.sortBy = filters.sortBy;
      if (filters.order) params.order = filters.order;
      if (filters.type) params.type = filters.type;
      if (filters.category) params.category = filters.category;
      if (filters.minRevenue) params.minRevenue = filters.minRevenue;
      if (filters.maxRevenue) params.maxRevenue = filters.maxRevenue;

      const response = await marketplaceAPI.getMarketplace(params);
      
      setStartups(response.data.results || []);
      setCurrentPage(1);
    } catch (error) {
      console.error('Failed to load startups:', error);
      setStartups([]);
    } finally {
      setLoading(false);
    }
  };

  const loadRecommendedStartups = async () => {
    try {
      setRecommendedLoading(true);
      const params = {
        limit: recommendedStartupsPerPage,
        offset: (recommendedCurrentPage - 1) * recommendedStartupsPerPage
      };
      
      // Add category filter if present
      if (filters.category && filters.category.trim()) {
        params.category = filters.category;
      }
      
      const response = await recommendationAPI.getPersonalizedStartups(params);
      
      console.log('[Marketplace] Recommended startups response:', response.data);
      
      if (response.data.error) {
        toast.warning('Recommendation service temporarily unavailable. Switching to all listings.');
        setViewMode('all');
        return;
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

  const [showFilters, setShowFilters] = useState(false);

  const categories = [
    { value: '', label: 'All Categories' },
    { value: 'saas', label: 'SaaS' },
    { value: 'ecommerce', label: 'Ecommerce' },
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

  const toggleFilters = () => {
    setShowFilters((prev) => !prev);
  };

  const startIndex = (currentPage - 1) * startupsPerPage;
  const paginatedStartups = startups.slice(startIndex, startIndex + startupsPerPage);
  const totalPages = Math.ceil(startups.length / startupsPerPage);

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
          <h2>{viewMode === 'recommended' ? 'Personalized Startups for You' : 'Explore Marketplace'}</h2>

          {/* Tab System - Only show for investors */}
          {isInvestor() && (
            <div className={styles.tabContainer}>
              <button
                className={`${styles.tab} ${viewMode === 'recommended' ? styles.activeTab : ''}`}
                onClick={() => {
                  setViewMode('recommended');
                  setRecommendedCurrentPage(1);
                }}
              >
                Personalized for You
              </button>
              <button
                className={`${styles.tab} ${viewMode === 'all' ? styles.activeTab : ''}`}
                onClick={() => {
                  setViewMode('all');
                  setCurrentPage(1);
                }}
              >
                All Listings
              </button>
            </div>
          )}

          {/* Mobile Filter Toggle Button - Only show in 'all' mode */}
          {viewMode === 'all' && (
            <button className={styles.filterToggle} onClick={toggleFilters}>
              {showFilters ? "Hide Filters" : "Show Filters"}
            </button>
          )}

          {/* Filters Section - Only show in 'all' mode or category filter for recommended */}
          {viewMode === 'all' ? (
            <div
              className={`${styles.filters} ${
                showFilters ? styles.showFilters : ""
              }`}
            >
              <div className={styles.filterRow}>
                <select 
                  value={filters.sortBy} 
                  onChange={(e) => handleFilterChange('sortBy', e.target.value)}
                >
                  <option value="date">Sort by Date</option>
                  <option value="price">Sort by Price</option>
                  <option value="revenue">Sort by Revenue</option>
                </select>
                
                <select 
                  value={filters.order} 
                  onChange={(e) => handleFilterChange('order', e.target.value)}
                >
                  <option value="desc">Newest First</option>
                  <option value="asc">Oldest First</option>
                </select>
                
                <select 
                  value={filters.category} 
                  onChange={(e) => handleFilterChange('category', e.target.value)}
                >
                  {categories.map(cat => (
                    <option key={cat.value} value={cat.value}>
                      {cat.label}
                    </option>
                  ))}
                </select>
              </div>
            </div>
          ) : (
            // Category filter for recommended mode
            <div className={styles.filters}>
              <div className={styles.filterRow}>
                <select 
                  value={filters.category} 
                  onChange={(e) => handleFilterChange('category', e.target.value)}
                >
                  {categories.map(cat => (
                    <option key={cat.value} value={cat.value}>
                      {cat.label}
                    </option>
                  ))}
                </select>
              </div>
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
                
                return (
                  <MarketPlaceCard
                    key={startup.id || index}
                    {...startup}
                    score={score}
                    matchReasons={reasons}
                  />
                );
              })
            ) : (
              <div className={styles.noResults}>
                <h3>No recommendations available</h3>
                <p>We're building your personalized recommendations. Try the "All Listings" tab to browse all available startups.</p>
              </div>
            )}
          </div>
        ) : (
          // All Listings View
          <div className={styles.cardsGrid}>
            {loading ? (
              <div className={styles.loading}>Loading startups...</div>
            ) : startups.length > 0 ? (
              paginatedStartups.map((startup, index) => (
                <MarketPlaceCard key={startup.id || index} {...startup} />
              ))
            ) : (
              <div className={styles.noResults}>
                <h3>No startups found</h3>
                <p>Try adjusting your filters or check back later for new listings.</p>
              </div>
            )}
          </div>
        )}
        
        {/* Pagination for Recommended Mode */}
        {viewMode === 'recommended' && totalRecommended > recommendedStartupsPerPage && (
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
            <span>Page {recommendedCurrentPage} of {Math.ceil(totalRecommended / recommendedStartupsPerPage)}</span>
            <button
              type="button"
              onClick={() => {
                setRecommendedCurrentPage(p => Math.min(Math.ceil(totalRecommended / recommendedStartupsPerPage), p + 1));
                window.scrollTo({ top: 0, behavior: 'smooth' });
              }}
              disabled={recommendedCurrentPage >= Math.ceil(totalRecommended / recommendedStartupsPerPage)}
            >
              &rarr;
            </button>
          </div>
        )}
        
        {/* Pagination for All Mode */}
        {viewMode === 'all' && startups.length > 0 && totalPages > 1 && (
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

export { Marketplace };
