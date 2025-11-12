import React, { useState, useEffect } from "react";
import styles from "./StartupDetails.module.css";
import { Navbar } from "../../components/Navbar/Navbar";
import { Footer } from "../../components/Footer/Footer";
import { Link, useParams, useNavigate } from "react-router-dom";
import { toast } from 'react-toastify';
import { useAuth } from '../../contexts/AuthContext';
import { startupAPI, userAPI, recommendationAPI } from '../../utils/apiServices';
import apiClient from '../../utils/axiosConfig';
import { BarChart, LineChart, Bar, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ReferenceLine, ResponsiveContainer, LabelList, Label } from 'recharts';

const StartupDetails = () => {
  const { id } = useParams();
  const navigate = useNavigate();
  const { user, isStudent, isInvestor, isEntrepreneur } = useAuth();
  const [startup, setStartup] = useState(null);
  const [positions, setPositions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [isFavorited, setIsFavorited] = useState(false);
  const [hasInterest, setHasInterest] = useState(false);
  const [interestMessage, setInterestMessage] = useState('');
  const [hasLike, setHasLike] = useState(false);
  const [hasDislike, setHasDislike] = useState(false);
  const [appliedPositionIds, setAppliedPositionIds] = useState(new Set());
  const [appliedStartupIds, setAppliedStartupIds] = useState(new Set());
  const [showConvertModal, setShowConvertModal] = useState(false);
  const [converting, setConverting] = useState(false);
  const [convertFormData, setConvertFormData] = useState({
    revenue: '',
    profit: '',
    asking_price: '',
    ttm_revenue: '',
    ttm_profit: '',
    last_month_revenue: '',
    last_month_profit: ''
  });

  useEffect(() => {
    loadStartupDetails();
    loadPositions();
    checkFavoriteStatus();
    checkInterestStatus();
    checkAppliedPositions();
    checkInteractionStatus();
  }, [id]);

  const loadStartupDetails = async () => {
    try {
      const response = await startupAPI.getStartup(id);
      setStartup(response.data);
    } catch (error) {
      console.error('Failed to load startup details:', error);
      toast.error('Failed to load startup details');
      navigate('/marketplace');
    } finally {
      setLoading(false);
    }
  };

  const loadPositions = async () => {
    try {
      const response = await startupAPI.getStartupPositions(id);
      // Handle both response formats
      if (response.data.positions) {
        setPositions(response.data.positions.filter(pos => pos.is_active));
      } else if (Array.isArray(response.data)) {
        setPositions(response.data.filter(pos => pos.is_active));
      }
    } catch (error) {
      console.error('Failed to load positions:', error);
      // Don't show error for this, positions might not exist
      setPositions([]);
    }
  };

  const checkFavoriteStatus = async () => {
    if (!user || !isInvestor()) return;
    
    try {
      const response = await userAPI.getUserFavorites();
      console.log('[StartupDetails] Checking favorite status for startup:', id);
      console.log('[StartupDetails] Favorites response:', response.data);
      
      // Handle both paginated and non-paginated responses
      const favorites = response.data.results || response.data;
      const isFav = Array.isArray(favorites) && favorites.some(fav => fav.startup.id === id);
      
      console.log('[StartupDetails] Is favorited:', isFav);
      setIsFavorited(isFav);
    } catch (error) {
      console.error('Failed to check favorite status:', error);
      // Don't show error to user for this non-critical operation
    }
  };

  const checkInterestStatus = async () => {
    if (!user || !isInvestor()) return;
    
    try {
      const response = await userAPI.getUserInterests();
      console.log('[StartupDetails] Checking interest status for startup:', id);
      console.log('[StartupDetails] Interests response:', response.data);
      
      // Handle both paginated and non-paginated responses
      const interests = response.data.results || response.data;
      const hasInt = Array.isArray(interests) && interests.some(interest => interest.startup.id === id);
      
      console.log('[StartupDetails] Has interest:', hasInt);
      setHasInterest(hasInt);
    } catch (error) {
      console.error('Failed to check interest status:', error);
      // Don't show error to user for this non-critical operation
    }
  };

  const checkInteractionStatus = async () => {
    if (!user) return;
    
    try {
      const response = await recommendationAPI.getStartupInteractionStatus(id);
      setHasLike(response.data.has_like || false);
      setHasDislike(response.data.has_dislike || false);
    } catch (error) {
      console.error('Failed to check interaction status:', error);
    }
  };

  const handleLike = async () => {
    if (!user) {
      toast.error('Please log in to like startups');
      navigate('/login');
      return;
    }

    try {
      if (hasLike) {
        await recommendationAPI.unlikeStartup(id);
        setHasLike(false);
        toast.success('Like removed');
      } else {
        await recommendationAPI.likeStartup(id);
        setHasLike(true);
        setHasDislike(false);
        toast.success('Startup liked!');
      }
    } catch (error) {
      console.error('Failed to like startup:', error);
      toast.error('Failed to like startup');
    }
  };

  const handleDislike = async () => {
    if (!user) {
      toast.error('Please log in to dislike startups');
      navigate('/login');
      return;
    }

    try {
      if (hasDislike) {
        await recommendationAPI.undislikeStartup(id);
        setHasDislike(false);
        toast.success('Dislike removed');
      } else {
        await recommendationAPI.dislikeStartup(id);
        setHasDislike(true);
        setHasLike(false);
        toast.success('Startup disliked');
      }
    } catch (error) {
      console.error('Failed to dislike startup:', error);
      toast.error('Failed to dislike startup');
    }
  };

  const checkAppliedPositions = async () => {
    // Also check which startups the user has applied to (for one-application-per-startup rule)
    if (!user || isInvestor()) return;
    
    try {
      const response = await userAPI.getUserApplications();
      console.log('[StartupDetails] Checking applied positions and startups');
      
      const applications = response.data.results || response.data;
      
      if (Array.isArray(applications)) {
        // Extract position IDs
        const appliedIds = new Set(
          applications
            .filter(app => app.position && app.position.id)
            .map(app => app.position.id)
        );
        
        // Extract startup IDs (to check if user has applied to any position in this startup)
        const appliedStartupIdsSet = new Set(
          applications
            .filter(app => app.startup && app.startup.id)
            .map(app => app.startup.id)
        );
        
        console.log('[StartupDetails] Applied position IDs:', Array.from(appliedIds));
        console.log('[StartupDetails] Applied startup IDs:', Array.from(appliedStartupIdsSet));
        setAppliedPositionIds(appliedIds);
        setAppliedStartupIds(appliedStartupIdsSet);
      }
    } catch (error) {
      console.error('Failed to check applied positions:', error);
    }
  };

  const toggleFavorite = async () => {
    // Check if user is authenticated
    if (!user || !isInvestor()) {
      toast.error('Please log in as an investor to use favorites');
      navigate('/login');
      return;
    }

    console.log('Toggle Favorite - Current State:', isFavorited);
    console.log('Toggle Favorite - Startup ID:', id);

    try {
      if (isFavorited) {
        // Remove from favorites using DELETE
        console.log('Removing from favorites...');
        await apiClient.delete(`/api/startups/${id}/favorite`);
        setIsFavorited(false);
        console.log('Successfully removed from favorites');
        toast.success('Removed from favorites');
      } else {
        // Add to favorites using POST
        console.log('Adding to favorites...');
      await startupAPI.toggleFavorite(id);
        setIsFavorited(true);
        console.log('Successfully added to favorites');
        toast.success('Added to favorites');
      }
    } catch (error) {
      console.error('Failed to toggle favorite:', error);
      console.error('Error response:', error.response?.data);
      
      // Handle specific authentication errors
      if (error.response?.status === 403 || error.response?.status === 401) {
        toast.error('Please log in to use favorites');
        navigate('/login');
      } else {
        toast.error('Failed to update favorites');
      }
    }
  };

  const expressInterest = async () => {
    // Check if user is authenticated
    if (!user || !isInvestor()) {
      toast.error('Please log in as an investor to express interest');
      navigate('/login');
      return;
    }

    if (!interestMessage.trim()) {
      toast.error('Please enter a message');
      return;
    }

    try {
      await startupAPI.expressInterest(id, { message: interestMessage });
      setHasInterest(true);
      setInterestMessage('');
      toast.success('Interest expressed successfully!');
    } catch (error) {
      console.error('Failed to express interest:', error);
      
      // Handle specific authentication errors
      if (error.response?.status === 403 || error.response?.status === 401) {
        toast.error('Please log in to express interest');
        navigate('/login');
      } else {
        toast.error('Failed to express interest');
      }
    }
  };

  const handleConvertToMarketplace = async (e) => {
    e.preventDefault();
    setConverting(true);

    try {
      await startupAPI.convertToMarketplace(id, convertFormData);
      toast.success('Startup successfully converted to Marketplace!');
      setShowConvertModal(false);
      // Reload startup details to reflect the change
      await loadStartupDetails();
    } catch (error) {
      console.error('Failed to convert startup:', error);
      const errorMessage = error.response?.data?.detail || error.response?.data?.error || 'Failed to convert startup';
      toast.error(errorMessage);
    } finally {
      setConverting(false);
    }
  };

  // Helper function to parse currency values to numbers
  const parseCurrency = (value) => {
    if (!value) return 0;
    // Remove $, commas, and any whitespace, then parse to float
    const cleaned = String(value).replace(/[$,\s]/g, '').trim();
    const parsed = parseFloat(cleaned);
    return isNaN(parsed) ? 0 : parsed;
  };

  // Financial Charts Component
  const FinancialChart = ({ startup }) => {
    const revenueData = [
      {
        label: 'TTM',
        revenue: parseCurrency(startup.performance?.ttmRevenue || startup.ttm_revenue || 0)
      },
      {
        label: 'Last Month',
        revenue: parseCurrency(startup.performance?.lastMonthRevenue || startup.last_month_revenue || 0)
      },
      {
        label: 'Current',
        revenue: parseCurrency(startup.revenue || 0)
      }
    ];

    const profitData = [
      {
        label: 'TTM',
        profit: parseCurrency(startup.performance?.ttmProfit || startup.ttm_profit || 0)
      },
      {
        label: 'Last Month',
        profit: parseCurrency(startup.performance?.lastMonthProfit || startup.last_month_profit || 0)
      },
      {
        label: 'Current',
        profit: parseCurrency(startup.profit || 0)
      }
    ];

    const askingPrice = parseCurrency(startup.asking_price || 0);

    // Calculate max values for Y-axes
    const maxRevenue = Math.max(...revenueData.map(d => d.revenue));
    const maxProfit = Math.max(...profitData.map(d => d.profit));
    const revenueYAxisMax = maxRevenue > 0 ? Math.ceil(maxRevenue * 1.2) : 10000;
    const profitYAxisMax = Math.max(maxProfit, askingPrice) > 0 ? Math.ceil(Math.max(maxProfit, askingPrice) * 1.2) : 10000;

    return (
      <div className={styles.chartsWrapper}>
        {/* Revenue Bar Chart */}
        <div className={styles.chartSection}>
          <h4 className={styles.chartTitle}>Revenue Comparison</h4>
          <ResponsiveContainer width="100%" height={450}>
            <BarChart data={revenueData} margin={{ top: 30, right: 40, left: 30, bottom: 70 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="label" 
                tick={{ fontSize: 14, fill: '#374151', fontWeight: 600 }}
                label={{ value: 'Time Period', position: 'insideBottom', offset: -10, style: { textAnchor: 'middle', fontSize: 14, fill: '#374151', fontWeight: 700 } }}
              />
              <YAxis 
                tick={{ fontSize: 12, fill: '#374151', fontWeight: 600 }}
                label={{ value: 'Amount ($)', angle: -90, position: 'insideLeft', style: { textAnchor: 'middle', fontSize: 14, fill: '#374151', fontWeight: 700 } }}
                domain={[0, revenueYAxisMax]}
                width={80}
              />
              <Tooltip 
                formatter={(value) => [`$${value.toLocaleString()}`, 'Revenue']}
                labelFormatter={(label) => `Period: ${label}`}
                contentStyle={{ 
                  backgroundColor: 'rgba(255, 255, 255, 0.95)', 
                  border: '1px solid #e5e7eb',
                  borderRadius: '8px',
                  fontSize: '14px',
                  fontWeight: 600
                }}
              />
              <Legend wrapperStyle={{ paddingTop: '20px', fontWeight: 600 }} />
              <Bar 
                dataKey="revenue" 
                fill="#2563eb" 
                name="Revenue"
                radius={[8, 8, 0, 0]}
              >
                <LabelList 
                  dataKey="revenue" 
                  position="top" 
                  formatter={(value) => `$${value.toLocaleString()}`}
                  style={{ fill: '#1f2937', fontSize: '12px', fontWeight: 600 }}
                />
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Profit Line Chart */}
        <div className={styles.chartSection}>
          <h4 className={styles.chartTitle}>Profit Comparison</h4>
          <ResponsiveContainer width="100%" height={450}>
            <LineChart data={profitData} margin={{ top: 30, right: 120, left: 50, bottom: 70 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="label" 
                tick={{ fontSize: 14, fill: '#374151', fontWeight: 600 }}
                label={{ value: 'Time Period', position: 'insideBottom', offset: -10, style: { textAnchor: 'middle', fontSize: 14, fill: '#374151', fontWeight: 700 } }}
              />
              <YAxis 
                tick={{ fontSize: 12, fill: '#374151', fontWeight: 600 }}
                label={{ value: 'Amount ($)', angle: -90, position: 'insideLeft', style: { textAnchor: 'middle', fontSize: 14, fill: '#374151', fontWeight: 700 } }}
                domain={[0, profitYAxisMax]}
                width={80}
              />
              <Tooltip 
                formatter={(value) => [`$${value.toLocaleString()}`, 'Profit']}
                labelFormatter={(label) => `Period: ${label}`}
                contentStyle={{ 
                  backgroundColor: 'rgba(255, 255, 255, 0.95)', 
                  border: '1px solid #e5e7eb',
                  borderRadius: '8px',
                  fontSize: '14px',
                  fontWeight: 600
                }}
              />
              <Legend wrapperStyle={{ paddingTop: '20px', fontWeight: 600 }} />
              <Line 
                type="monotone" 
                dataKey="profit" 
                stroke="#f97316" 
                strokeWidth={3}
                dot={{ r: 8, fill: '#f97316', strokeWidth: 2, stroke: '#fff' }}
                activeDot={{ r: 10 }}
                name="Profit"
              >
                <LabelList 
                  dataKey="profit" 
                  position="top" 
                  formatter={(value) => `$${value.toLocaleString()}`}
                  style={{ fill: '#1f2937', fontSize: '12px', fontWeight: 600 }}
                  offset={10}
                />
              </Line>
              {askingPrice > 0 && (
                <ReferenceLine 
                  y={askingPrice} 
                  stroke="#6b7280"
                  strokeDasharray="5 5"
                  strokeWidth={2}
                >
                  <Label 
                    value={`$${askingPrice.toLocaleString()}`}
                    position="left"
                    offset={15}
                    style={{ 
                      fontSize: 13, 
                      fill: '#000000', 
                      fontWeight: 700,
                      backgroundColor: 'rgba(255, 255, 255, 0.98)',
                      padding: '5px 10px',
                      borderRadius: '6px',
                      border: '2px solid #6b7280',
                      boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
                    }}
                  />
                  <Label 
                    value="Asking Price"
                    position="right"
                    offset={20}
                    style={{ 
                      fontSize: 13, 
                      fill: '#6b7280', 
                      fontWeight: 600, 
                      backgroundColor: 'rgba(255, 255, 255, 0.95)', 
                      padding: '4px 8px', 
                      borderRadius: '4px', 
                      border: '1px solid #e5e7eb' 
                    }}
                  />
                </ReferenceLine>
              )}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    );
  };

  if (loading) {
    return (
      <>
        <Navbar/>
        <div className={styles.container}>
          <div className={styles.loading}>Loading startup details...</div>
        </div>
        <Footer/>
      </>
    );
  }

  if (!startup) {
    return (
      <>
        <Navbar/>
        <div className={styles.container}>
          <div className={styles.error}>Startup not found</div>
        </div>
        <Footer/>
      </>
    );
  }

  return (
    <>
    <Navbar/>
    <div className={styles.container}>
      {/* Header with Title and Action Buttons */}
      <div className={styles.headerSection}>
      <h1 className={styles.title}>{startup.title}</h1>
        
        <div className={styles.actionButtons}>
          {/* Like/Dislike Buttons (for all logged-in users) */}
          {user && (
            <>
              <button 
                className={`${styles.actionBtn} ${hasLike ? styles.liked : ''}`}
                onClick={handleLike}
                title={hasLike ? 'Remove like' : 'Like this startup'}
              >
                {hasLike ? '👍 Liked' : '👍 Like'}
              </button>
              <button 
                className={`${styles.actionBtn} ${hasDislike ? styles.disliked : ''}`}
                onClick={handleDislike}
                title={hasDislike ? 'Remove dislike' : 'Dislike this startup'}
              >
                {hasDislike ? '👎 Disliked' : '👎 Dislike'}
              </button>
            </>
          )}
          
          {/* Investor Favorite Button */}
          {isInvestor() && (
            <button 
              className={`${styles.actionBtn} ${isFavorited ? styles.favorited : ''}`}
              onClick={toggleFavorite}
            >
              {isFavorited ? '❤️ Favorited' : '🤍 Add to Favorites'}
            </button>
          )}
        </div>
      </div>

      {/* Tags */}
      <div className={styles.tags}>
        <span className={`${styles.tag} ${styles.fund}`}>{startup.category}</span>
        <span className={`${styles.tag} ${styles.equity}`}>{startup.type}</span>
        <span className={`${styles.tag} ${styles.collab}`}>{startup.field}</span>
      </div>

      {/* Description */}
      <h3 className={styles.sectionTitle}>Description</h3>
      <p className={styles.description}>
        {startup.description}
      </p>

      <hr className={styles.divider} />

      {/* Conditional Performance Section based on Type */}
      {startup.type === 'marketplace' ? (
        <>
          <h3 className={styles.sectionTitle}>Financial Performance</h3>
      <div className={styles.performance}>
        <div className={styles.metric}>
          <span className={styles.metricLabel}>TTM REVENUE</span>
          <div className={styles.metricLabel2}>
                <img src="/Get Revenue.svg" alt="" />
            <p className={styles.metricValue}>{startup.performance?.ttmRevenue || startup.ttm_revenue || '$0'}</p>
          </div>
        </div>
        <div className={styles.metric}>
          <span className={styles.metricLabel}>TTM PROFIT</span>
          <div className={styles.metricLabel2}>
                <img src="/Stocks Growth.svg" alt="" />
            <p className={styles.metricValue}>{startup.performance?.ttmProfit || startup.ttm_profit || '$0'}</p>
          </div>
        </div>
        <div className={styles.metric}>
          <span className={styles.metricLabel}>LAST MONTH REVENUE</span>
          <div className={styles.metricLabel2}>
                <img src="/Profit.svg" alt="" />
            <p className={styles.metricValue}>{startup.performance?.lastMonthRevenue || startup.last_month_revenue || '$0'}</p>
          </div>
        </div>
        <div className={styles.metric}>
          <span className={styles.metricLabel}>LAST MONTH PROFIT</span>
          <div className={styles.metricLabel2}>
                <img src="/Weak Financial Growth.svg" alt="" />
            <p className={styles.metricValue}>{startup.performance?.lastMonthProfit || startup.last_month_profit || '$0'}</p>
          </div>
        </div>
      </div>
          
          <div className={styles.performance}>
            <div className={styles.metric}>
              <span className={styles.metricLabel}>CURRENT REVENUE</span>
              <p className={styles.metricValue}>{startup.revenue || '$0'}</p>
            </div>
            <div className={styles.metric}>
              <span className={styles.metricLabel}>CURRENT PROFIT</span>
              <p className={styles.metricValue}>{startup.profit || '$0'}</p>
            </div>
            <div className={styles.metric}>
              <span className={styles.metricLabel}>ASKING PRICE</span>
              <p className={styles.metricValue}>{startup.asking_price || '$0'}</p>
            </div>
          </div>

          {/* Financial Performance Chart */}
          <div className={styles.chartContainer}>
            <h3 className={styles.sectionTitle}>Financial Performance Chart</h3>
            <FinancialChart startup={startup} />
          </div>
        </>
      ) : (
        <>
          <h3 className={styles.sectionTitle}>Collaboration Details</h3>
          <div className={styles.performance}>
            <div className={styles.metric}>
              <span className={styles.metricLabel}>EARN THROUGH</span>
              <p className={styles.metricValue}>{startup.earn_through || 'N/A'}</p>
            </div>
            <div className={styles.metric}>
              <span className={styles.metricLabel}>CURRENT PHASE</span>
              <p className={styles.metricValue}>{startup.phase || 'N/A'}</p>
            </div>
            <div className={styles.metric}>
              <span className={styles.metricLabel}>TEAM SIZE</span>
              <p className={styles.metricValue}>{startup.team_size || 'N/A'}</p>
            </div>
            <div className={styles.metric}>
              <span className={styles.metricLabel}>CATEGORY</span>
              <p className={styles.metricValue}>
                {startup.category ? startup.category.charAt(0).toUpperCase() + startup.category.slice(1) : 'N/A'}
              </p>
            </div>
          </div>
        </>
      )}

      <hr className={styles.divider} />

      {/* Positions - Only for Collaboration startups, Hidden for Investors */}
      {!isInvestor() && startup?.type === 'collaboration' && (
        <>
      <h3 className={styles.sectionTitle}>Available Positions</h3>
      <div className={styles.positions}>
        {positions.length > 0 ? (
          positions.map((position) => (
            <div key={position.id} className={styles.positionCard}>
              <h4 className={styles.positionTitle}>{position.title}</h4>
              <p className={styles.positionDescription}>{position.description}</p>
              {position.requirements && (
                <p className={styles.positionRequirements}>
                  <strong>Requirements:</strong> {position.requirements}
                </p>
              )}
              {/* Show Apply button for students OR entrepreneurs who don't own this startup */}
              {(isStudent() || (isEntrepreneur() && user?.id !== startup?.owner?.id)) && (
                appliedPositionIds.has(position.id) || appliedStartupIds.has(startup?.id) ? (
                  <div className={styles.appliedBadge}>
                    {appliedStartupIds.has(startup?.id) && !appliedPositionIds.has(position.id) ? (
                      'You have already applied to one position for this startup'
                    ) : (
                      'Applied'
                    )}
                  </div>
                ) : (
                  <Link 
                    to={`/apply-for-collaboration/${id}?position=${position.id}`} 
                    className={styles.applyButton}
                  >
                    Apply for this Position
                  </Link>
                )
              )}
            </div>
          ))
        ) : (
          <div className={styles.noPositions}>
            {startup?.type === 'collaboration' 
              ? 'No open positions at the moment. Check back later!' 
              : 'This startup is not currently looking for team members.'}
          </div>
        )}
      </div>
      
      {/* Entrepreneur Actions - Only for Collaboration startups */}
      {isEntrepreneur() && user?.id === startup?.owner?.id && startup?.type === 'collaboration' && (
        <div className={styles.entrepreneurActions}>
          <Link to={`/startups/${id}/positions`} className={styles.manageButton}>
            Manage Positions
          </Link>
            </div>
          )}
        </>
      )}
      
      {/* Investor Express Interest Section - At End */}
      {isInvestor() && (
        <div className={styles.investorActions}>
          {!hasInterest && (
            <div className={styles.interestSection}>
              <h3 className={styles.sectionTitle}>Express Your Interest</h3>
              <textarea
                value={interestMessage}
                onChange={(e) => setInterestMessage(e.target.value)}
                placeholder="Express your interest in this startup..."
                className={styles.interestInput}
                rows="3"
              />
              <button 
                className={styles.actionBtn}
                onClick={expressInterest}
              >
                Express Interest
              </button>
            </div>
          )}
          
          {hasInterest && (
            <div className={styles.interestStatus}>
              ✅ Interest expressed
            </div>
          )}
        </div>
      )}

      {/* Convert to Marketplace Button - Only for Collaboration startups owned by entrepreneur */}
      {isEntrepreneur() && user?.id === startup?.owner?.id && startup?.type === 'collaboration' && (
        <div className={styles.entrepreneurActions}>
          <button 
            className={styles.convertButton}
            onClick={() => setShowConvertModal(true)}
          >
            Convert to Marketplace
          </button>
        </div>
      )}

      {/* Convert to Marketplace Modal */}
      {showConvertModal && (
        <div className={styles.modalOverlay} onClick={() => !converting && setShowConvertModal(false)}>
          <div className={styles.modalContent} onClick={(e) => e.stopPropagation()}>
            <h2 className={styles.modalTitle}>Convert Startup to Marketplace</h2>
            
            <div className={styles.warningMessage}>
              <strong>⚠️ Important:</strong> From now on, you will not be able to manage positions for this startup. 
              All existing positions will remain visible but cannot be managed or created after conversion.
            </div>

            <form onSubmit={handleConvertToMarketplace} className={styles.convertForm}>
              <h4 className={styles.sectionSubtitle}>Financial Information (Required)</h4>
              
              <div className={styles.formRow}>
                <div className={styles.formGroup}>
                  <label>Current Revenue *</label>
                  <input 
                    type="text" 
                    name="revenue"
                    value={convertFormData.revenue}
                    onChange={(e) => setConvertFormData({...convertFormData, revenue: e.target.value})}
                    placeholder="e.g., $10,000/month" 
                    required
                    disabled={converting}
                  />
                </div>
                <div className={styles.formGroup}>
                  <label>Current Profit *</label>
                  <input 
                    type="text" 
                    name="profit"
                    value={convertFormData.profit}
                    onChange={(e) => setConvertFormData({...convertFormData, profit: e.target.value})}
                    placeholder="e.g., $5,000/month" 
                    required
                    disabled={converting}
                  />
                </div>
              </div>

              <div className={styles.formRow}>
                <div className={styles.formGroup}>
                  <label>Asking Price *</label>
                  <input 
                    type="text" 
                    name="asking_price"
                    value={convertFormData.asking_price}
                    onChange={(e) => setConvertFormData({...convertFormData, asking_price: e.target.value})}
                    placeholder="e.g., $100,000" 
                    required
                    disabled={converting}
                  />
                </div>
                <div className={styles.formGroup}>
                  <label>TTM Revenue *</label>
                  <input 
                    type="text" 
                    name="ttm_revenue"
                    value={convertFormData.ttm_revenue}
                    onChange={(e) => setConvertFormData({...convertFormData, ttm_revenue: e.target.value})}
                    placeholder="e.g., $120,000" 
                    required
                    disabled={converting}
                  />
                </div>
              </div>

              <div className={styles.formRow}>
                <div className={styles.formGroup}>
                  <label>TTM Profit *</label>
                  <input 
                    type="text" 
                    name="ttm_profit"
                    value={convertFormData.ttm_profit}
                    onChange={(e) => setConvertFormData({...convertFormData, ttm_profit: e.target.value})}
                    placeholder="e.g., $60,000" 
                    required
                    disabled={converting}
                  />
                </div>
                <div className={styles.formGroup}>
                  <label>Last Month Revenue *</label>
                  <input 
                    type="text" 
                    name="last_month_revenue"
                    value={convertFormData.last_month_revenue}
                    onChange={(e) => setConvertFormData({...convertFormData, last_month_revenue: e.target.value})}
                    placeholder="e.g., $12,000" 
                    required
                    disabled={converting}
                  />
                </div>
              </div>

              <div className={styles.formRow}>
                <div className={styles.formGroup}>
                  <label>Last Month Profit *</label>
                  <input 
                    type="text" 
                    name="last_month_profit"
                    value={convertFormData.last_month_profit}
                    onChange={(e) => setConvertFormData({...convertFormData, last_month_profit: e.target.value})}
                    placeholder="e.g., $6,000" 
                    required
                    disabled={converting}
                  />
                </div>
              </div>

              <div className={styles.modalActions}>
                <button 
                  type="button"
                  className={styles.cancelButton}
                  onClick={() => setShowConvertModal(false)}
                  disabled={converting}
                >
                  Cancel
                </button>
                <button 
                  type="submit"
                  className={styles.confirmButton}
                  disabled={converting}
                >
                  {converting ? 'Converting...' : 'Confirm Conversion'}
                </button>
              </div>
            </form>

            {/* Circular Progress Indicator */}
            {converting && (
              <div className={styles.progressOverlay}>
                <div className={styles.progressModal}>
                  <div className={styles.progressSpinner}></div>
                  <p className={styles.progressText}>Converting to Marketplace...</p>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>

    <Footer/>
    </>
  );
};

export default StartupDetails;
