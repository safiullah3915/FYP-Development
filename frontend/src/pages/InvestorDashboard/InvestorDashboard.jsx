import React, { useState, useEffect } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { Navbar } from '../../components/Navbar/Navbar';
import { Footer } from '../../components/Footer/Footer';
import MarketPlaceCard from '../../components/MarketplaceCard/MarketPlaceCard';
import styles from './InvestorDashboard.module.css';
import { userAPI, startupAPI, marketplaceAPI } from '../../utils/apiServices';
import { useAuth } from '../../contexts/AuthContext';
import { toast } from 'react-toastify';

const InvestorDashboard = () => {
  const { user } = useAuth();
  const [searchParams] = useSearchParams();
  const tabFromUrl = searchParams.get('tab') || 'favorites';
  const [activeTab, setActiveTab] = useState(tabFromUrl);
  const [favorites, setFavorites] = useState([]);
  const [interests, setInterests] = useState([]);
  const [recommendedStartups, setRecommendedStartups] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Update active tab when URL changes
    const tab = searchParams.get('tab') || 'favorites';
    setActiveTab(tab);
  }, [searchParams]);

  useEffect(() => {
    loadDashboardData();
  }, []);

  const loadDashboardData = async () => {
    try {
      setLoading(true);
      console.log('[InvestorDashboard] Loading dashboard data...');
      
      const [favoritesRes, interestsRes, marketplaceRes] = await Promise.all([
        userAPI.getUserFavorites(),
        userAPI.getUserInterests(), 
        marketplaceAPI.getMarketplace({ limit: 6 })
      ]);
      
      console.log('[InvestorDashboard] Favorites Response:', favoritesRes.data);
      console.log('[InvestorDashboard] Interests Response:', interestsRes.data);
      console.log('[InvestorDashboard] Marketplace Response:', marketplaceRes.data);
      
      const favoritesData = (favoritesRes.data && favoritesRes.data.results) ? favoritesRes.data.results : Array.isArray(favoritesRes.data) ? favoritesRes.data : [];
      const interestsData = (interestsRes.data && interestsRes.data.results) ? interestsRes.data.results : Array.isArray(interestsRes.data) ? interestsRes.data : [];
      
      console.log('[InvestorDashboard] Processed Favorites:', favoritesData);
      console.log('[InvestorDashboard] Favorites count:', favoritesData.length);
      
      setFavorites(favoritesData);
      setInterests(interestsData);
      setRecommendedStartups(marketplaceRes.data.results || []);
    } catch (error) {
      console.error('Failed to load investor dashboard:', error);
      console.error('Error details:', error.response?.data);
      toast.error('Failed to load dashboard data');
    } finally {
      setLoading(false);
    }
  };

  const handleExpressInterest = async (startupId, message = '') => {
    try {
      await startupAPI.expressInterest(startupId, { message });
      toast.success('Interest expressed successfully!');
      loadDashboardData(); // Reload to update the lists
    } catch (error) {
      console.error('Failed to express interest:', error);
      toast.error('Failed to express interest');
    }
  };

  const handleToggleFavorite = async (startupId, isFavorited = true) => {
    try {
      console.log('[InvestorDashboard] Toggling favorite for startup:', startupId, 'Current state:', isFavorited);
      
      if (isFavorited) {
        // Remove from favorites using DELETE
        await startupAPI.deleteFavorite(startupId);
        toast.success('Removed from favorites');
      } else {
        // Add to favorites using POST
      await startupAPI.toggleFavorite(startupId);
        toast.success('Added to favorites');
      }
      
      loadDashboardData(); // Reload to update the lists
    } catch (error) {
      console.error('Failed to toggle favorite:', error);
      console.error('Error details:', error.response?.data);
      toast.error('Failed to update favorites');
    }
  };

  const renderTabContent = () => {
    if (loading) {
      return <div className={styles.loading}>Loading...</div>;
    }

    switch (activeTab) {
      case 'favorites':
        return (
          <div className={styles.tabContent}>
            <div className={styles.tabHeader}>
              <h2>Your Favorites ({favorites.length})</h2>
              <p>Startups you've bookmarked for future reference</p>
            </div>
            
            {favorites.length === 0 ? (
              <div className={styles.emptyState}>
                <h3>No favorites yet</h3>
                <p>Browse the marketplace and save startups you're interested in</p>
                <Link to="/marketplace" className={styles.actionButton}>
                  Browse Marketplace
                </Link>
              </div>
            ) : (
              <div className={styles.startupsGrid}>
                {favorites.map((favorite) => {
                  console.log('[InvestorDashboard] Rendering favorite:', favorite);
                  console.log('[InvestorDashboard] Startup data:', favorite.startup);
                  return (
                  <MarketPlaceCard 
                    key={favorite.startup.id} 
                    startup={favorite.startup}
                      onToggleFavorite={() => handleToggleFavorite(favorite.startup.id, true)}
                    onExpressInterest={(message) => handleExpressInterest(favorite.startup.id, message)}
                  />
                  );
                })}
              </div>
            )}
          </div>
        );

      case 'interests':
        return (
          <div className={styles.tabContent}>
            <div className={styles.tabHeader}>
              <h2>Expressed Interests ({interests.length})</h2>
              <p>Startups you've shown interest in investing</p>
            </div>

            {interests.length === 0 ? (
              <div className={styles.emptyState}>
                <h3>No interests expressed yet</h3>
                <p>Start exploring startups and express your investment interest</p>
                <Link to="/marketplace" className={styles.actionButton}>
                  Find Investment Opportunities
                </Link>
              </div>
            ) : (
              <div className={styles.interestsList}>
                {interests.map((interest) => (
                  <div key={interest.id} className={styles.interestCard}>
                    <div className={styles.startupInfo}>
                      <h4>{interest.startup.title}</h4>
                      <p className={styles.category}>{interest.startup.category}</p>
                      <p className={styles.description}>
                        {interest.startup.description.substring(0, 150)}...
                      </p>
                    </div>
                    <div className={styles.interestInfo}>
                      <div className={styles.interestDate}>
                        <span>Interest expressed on</span>
                        <strong>{new Date(interest.created_at).toLocaleDateString()}</strong>
                      </div>
                      {interest.message && (
                        <div className={styles.interestMessage}>
                          <span>Your message:</span>
                          <p>"{interest.message}"</p>
                        </div>
                      )}
                      <Link 
                        to={`/startupdetail/${interest.startup.id}`}
                        className={styles.viewButton}
                      >
                        View Details
                      </Link>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        );

      case 'recommended':
        return (
          <div className={styles.tabContent}>
            <div className={styles.tabHeader}>
              <h2>Investment Opportunities</h2>
              <p>Discover promising startups looking for funding</p>
            </div>

            {recommendedStartups.length === 0 ? (
              <div className={styles.emptyState}>
                <h3>No startups available</h3>
                <p>Check back later for new investment opportunities</p>
              </div>
            ) : (
              <div className={styles.startupsGrid}>
                {recommendedStartups.map((startup) => (
                  <MarketPlaceCard 
                    key={startup.id} 
                    startup={startup}
                    onToggleFavorite={() => handleToggleFavorite(startup.id)}
                    onExpressInterest={(message) => handleExpressInterest(startup.id, message)}
                  />
                ))}
              </div>
            )}
            
            <div className={styles.seeMore}>
              <Link to="/marketplace" className={styles.seeMoreButton}>
                View All Opportunities
              </Link>
            </div>
          </div>
        );

      default:
        return null;
    }
  };

  return (
    <>
      <Navbar />
      <div className={styles.container}>
        <div className={styles.header}>
          <h1>Investor Dashboard</h1>
          <p>Discover and invest in promising startups</p>
        </div>

        <div className={styles.statsSection}>
          <div className={styles.statCard}>
            <div className={styles.statNumber}>{favorites.length}</div>
            <div className={styles.statLabel}>Favorites</div>
          </div>
          <div className={styles.statCard}>
            <div className={styles.statNumber}>{interests.length}</div>
            <div className={styles.statLabel}>Interests Expressed</div>
          </div>
          <div className={styles.statCard}>
            <div className={styles.statNumber}>{recommendedStartups.length}+</div>
            <div className={styles.statLabel}>Available Opportunities</div>
          </div>
        </div>

        <div className={styles.tabNavigation}>
          <button
            className={`${styles.tabButton} ${activeTab === 'favorites' ? styles.active : ''}`}
            onClick={() => setActiveTab('favorites')}
          >
            My Favorites
          </button>
          <button
            className={`${styles.tabButton} ${activeTab === 'interests' ? styles.active : ''}`}
            onClick={() => setActiveTab('interests')}
          >
            My Interests
          </button>
          <button
            className={`${styles.tabButton} ${activeTab === 'recommended' ? styles.active : ''}`}
            onClick={() => setActiveTab('recommended')}
          >
            Investment Opportunities
          </button>
        </div>

        {renderTabContent()}

        <div className={styles.actionSection}>
          <div className={styles.actionCard}>
            <h3>Looking for specific opportunities?</h3>
            <p>Use our advanced search to find startups that match your investment criteria</p>
            <Link to="/search" className={styles.actionButton}>
              Advanced Search
            </Link>
          </div>
          <div className={styles.actionCard}>
            <h3>Connect with entrepreneurs</h3>
            <p>Send messages and build relationships with startup founders</p>
            <Link to="/message" className={styles.actionButton}>
              Messages
            </Link>
          </div>
        </div>
      </div>
      <Footer />
    </>
  );
};

export default InvestorDashboard;