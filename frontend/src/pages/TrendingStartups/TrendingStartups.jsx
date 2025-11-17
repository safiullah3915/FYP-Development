import React, { useState, useEffect } from "react";
import { Navbar } from '../../components/Navbar/Navbar';
import { Footer } from '../../components/Footer/Footer';
import TrendingStartupCard from '../../components/TrendingStartupCard/TrendingStartupCard';
import { recommendationAPI } from '../../utils/apiServices';
import { toast } from 'react-toastify';
import styles from "./TrendingStartups.module.css";

const TrendingStartups = () => {
  const [startups, setStartups] = useState([]);
  const [filteredStartups, setFilteredStartups] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Filter states
  const [sortBy, setSortBy] = useState('trending_score');
  const [categoryFilter, setCategoryFilter] = useState('');
  const [typeFilter, setTypeFilter] = useState('');
  const [searchQuery, setSearchQuery] = useState('');
  const [showFilters, setShowFilters] = useState(false);

  // Get unique categories and types from startups
  const categories = [...new Set(startups.map(s => s.category).filter(Boolean))].sort();
  const types = [...new Set(startups.map(s => s.type).filter(Boolean))].sort();

  useEffect(() => {
    loadTrendingStartups();
  }, []);

  useEffect(() => {
    applyFilters();
  }, [startups, sortBy, categoryFilter, typeFilter, searchQuery]);

  const loadTrendingStartups = async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await recommendationAPI.getTrendingStartups({
        limit: 10,
        sort_by: 'trending_score'
      });
      
      console.log('📊 [TrendingStartups] API Response:', response.data);
      console.log('📊 [TrendingStartups] Startups received:', response.data.startups?.length || 0);
      
      if (response.data.startups) {
        // Log first startup to verify ID is present
        if (response.data.startups.length > 0) {
          console.log('📊 [TrendingStartups] Sample startup:', response.data.startups[0]);
          console.log('📊 [TrendingStartups] Sample startup ID:', response.data.startups[0].id);
        }
        setStartups(response.data.startups);
      } else {
        setStartups([]);
        if (response.data.error) {
          setError(response.data.error);
          toast.warning('Recommendation service temporarily unavailable');
        }
      }
    } catch (error) {
      console.error('❌ [TrendingStartups] Failed to load trending startups:', error);
      setError('Failed to load trending startups');
      setStartups([]);
      toast.error('Failed to load trending startups');
    } finally {
      setLoading(false);
    }
  };

  const applyFilters = () => {
    let filtered = [...startups];

    // Search filter
    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      filtered = filtered.filter(startup => 
        startup.title?.toLowerCase().includes(query) ||
        startup.description?.toLowerCase().includes(query)
      );
    }

    // Category filter
    if (categoryFilter) {
      filtered = filtered.filter(startup => startup.category === categoryFilter);
    }

    // Type filter
    if (typeFilter) {
      filtered = filtered.filter(startup => startup.type === typeFilter);
    }

    // Sort
    filtered.sort((a, b) => {
      switch (sortBy) {
        case 'trending_score':
          return (b.trending_score || 0) - (a.trending_score || 0);
        case 'popularity_score':
          return (b.popularity_score || 0) - (a.popularity_score || 0);
        case 'created_at':
          return new Date(b.created_at || 0) - new Date(a.created_at || 0);
        case 'views':
          return (b.view_count_7d || 0) - (a.view_count_7d || 0);
        case 'velocity_score':
          return (b.velocity_score || 0) - (a.velocity_score || 0);
        default:
          return 0;
      }
    });

    setFilteredStartups(filtered);
  };

  const clearFilters = () => {
    setSortBy('trending_score');
    setCategoryFilter('');
    setTypeFilter('');
    setSearchQuery('');
  };

  const activeFiltersCount = [categoryFilter, typeFilter, searchQuery].filter(Boolean).length;

  if (loading) {
    return (
      <>
        <Navbar />
        <div className={styles.container}>
          <div className={styles.loadingContainer}>
            <div className={styles.spinner}></div>
            <p>Loading trending startups...</p>
          </div>
        </div>
        <Footer />
      </>
    );
  }

  return (
    <>
      <Navbar />
      <div className={styles.container}>
        {/* Header */}
        <div className={styles.header}>
          <div className={styles.headerContent}>
            <h1 className={styles.title}>
              <span className={styles.fireIcon}>🔥</span>
              Trending Startups
            </h1>
            <p className={styles.subtitle}>
              Discover the most popular and trending startups on the platform
            </p>
          </div>
        </div>

        {/* Filters Section */}
        <div className={styles.filtersSection}>
          <div className={styles.filtersHeader}>
            <button
              className={styles.filterToggle}
              onClick={() => setShowFilters(!showFilters)}
            >
              <span>🔍</span>
              Filters
              {activeFiltersCount > 0 && (
                <span className={styles.filterBadge}>{activeFiltersCount}</span>
              )}
            </button>
            {activeFiltersCount > 0 && (
              <button className={styles.clearFilters} onClick={clearFilters}>
                Clear All
              </button>
            )}
          </div>

          {showFilters && (
            <div className={styles.filtersPanel}>
              {/* Search */}
              <div className={styles.filterGroup}>
                <label className={styles.filterLabel}>Search</label>
                <input
                  type="text"
                  className={styles.searchInput}
                  placeholder="Search by title or description..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                />
              </div>

              {/* Sort By */}
              <div className={styles.filterGroup}>
                <label className={styles.filterLabel}>Sort By</label>
                <select
                  className={styles.filterSelect}
                  value={sortBy}
                  onChange={(e) => setSortBy(e.target.value)}
                >
                  <option value="trending_score">Trending Score</option>
                  <option value="popularity_score">Popularity Score</option>
                  <option value="velocity_score">Growth Velocity</option>
                  <option value="views">Views (7 days)</option>
                  <option value="created_at">Date Created</option>
                </select>
              </div>

              {/* Category Filter */}
              {categories.length > 0 && (
                <div className={styles.filterGroup}>
                  <label className={styles.filterLabel}>Category</label>
                  <select
                    className={styles.filterSelect}
                    value={categoryFilter}
                    onChange={(e) => setCategoryFilter(e.target.value)}
                  >
                    <option value="">All Categories</option>
                    {categories.map(category => (
                      <option key={category} value={category}>
                        {category.charAt(0).toUpperCase() + category.slice(1)}
                      </option>
                    ))}
                  </select>
                </div>
              )}

              {/* Type Filter */}
              {types.length > 0 && (
                <div className={styles.filterGroup}>
                  <label className={styles.filterLabel}>Type</label>
                  <select
                    className={styles.filterSelect}
                    value={typeFilter}
                    onChange={(e) => setTypeFilter(e.target.value)}
                  >
                    <option value="">All Types</option>
                    {types.map(type => (
                      <option key={type} value={type}>
                        {type.charAt(0).toUpperCase() + type.slice(1)}
                      </option>
                    ))}
                  </select>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Results */}
        {error && (
          <div className={styles.errorMessage}>
            <p>{error}</p>
          </div>
        )}

        {filteredStartups.length === 0 && !loading && !error ? (
          <div className={styles.emptyState}>
            <div className={styles.emptyIcon}>📊</div>
            <h2>No trending startups found</h2>
            <p>
              {activeFiltersCount > 0
                ? 'Try adjusting your filters to see more results.'
                : 'Check back later for trending startups.'}
            </p>
            {activeFiltersCount > 0 && (
              <button className={styles.clearFiltersButton} onClick={clearFilters}>
                Clear Filters
              </button>
            )}
          </div>
        ) : (
          <>
            <div className={styles.resultsHeader}>
              <p className={styles.resultsCount}>
                Showing {filteredStartups.length} of {startups.length} trending startups
              </p>
            </div>
            <div className={styles.startupsGrid}>
              {filteredStartups.map((startup) => (
                <TrendingStartupCard key={startup.id} startup={startup} />
              ))}
            </div>
          </>
        )}
      </div>
      <Footer />
    </>
  );
};

export default TrendingStartups;

