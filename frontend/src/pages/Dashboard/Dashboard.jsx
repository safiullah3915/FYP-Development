import React, { useState, useEffect, useRef } from 'react';
import { useAuth } from '../../contexts/AuthContext';
import { Navbar } from '../../components/Navbar/Navbar';
import { Footer } from '../../components/Footer/Footer';
import { Link } from 'react-router-dom';
import styles from './Dashboard.module.css';
import { userAPI, positionAPI } from '../../utils/apiServices';
import apiClient from '../../utils/axiosConfig';

const Dashboard = () => {
  const { user, isEntrepreneur, isStudent, isInvestor, loading: authLoading, isAuthenticated } = useAuth();
  const [stats, setStats] = useState(null);
  const [recentActivity, setRecentActivity] = useState([]);
  const [loading, setLoading] = useState(true);
  const [myStartups, setMyStartups] = useState([]);
  const [startupIdToPositions, setStartupIdToPositions] = useState({});
  const [currentPage, setCurrentPage] = useState(1);
  const applicationsPerPage = 3;
  const recentApplicationsRef = useRef(null);

  const formatDate = (value) => {
    if (!value) return '-';
    const parsed = new Date(value);
    if (Number.isNaN(parsed.getTime())) {
      return '-';
    }
    return parsed.toLocaleDateString();
  };

  const getStatusBadgeClass = (status) => {
    switch ((status || '').toLowerCase()) {
      case 'approved':
        return `${styles.statusBadge} ${styles.statusApproved}`;
      case 'rejected':
        return `${styles.statusBadge} ${styles.statusRejected}`;
      case 'withdrawn':
        return `${styles.statusBadge} ${styles.statusWithdrawn}`;
      default:
        return `${styles.statusBadge} ${styles.statusPending}`;
    }
  };

  const formatStatusLabel = (status) => {
    if (!status) {
      return 'Pending';
    }
    return status
      .replace(/_/g, ' ')
      .replace(/\b\w/g, (char) => char.toUpperCase());
  };

  const scrollToRecentApplications = () => {
    // Reset to first page to show most recent applications
    setCurrentPage(1);
    // Small delay to ensure state update, then scroll
    setTimeout(() => {
      if (recentApplicationsRef.current) {
        recentApplicationsRef.current.scrollIntoView({ 
          behavior: 'smooth', 
          block: 'start' 
        });
      }
    }, 100);
  };

  const totalPages = Math.ceil(recentActivity.length / applicationsPerPage);
  const displayedApplications = recentActivity.slice(
    (currentPage - 1) * applicationsPerPage,
    currentPage * applicationsPerPage
  );

  const handlePreviousPage = () => {
    setCurrentPage(prev => Math.max(1, prev - 1));
  };

  const handleNextPage = () => {
    setCurrentPage(prev => Math.min(totalPages, prev + 1));
  };

  useEffect(() => {
    // Load dashboard data when authentication is complete
    if (!authLoading && isAuthenticated) {
      if (user) {
        console.log('🏠 Dashboard: Auth complete, loading dashboard data...');
        loadDashboardData();
      } else {
        console.log('🏠 Dashboard: Authenticated but no user data yet, waiting...');
        // Set a timeout to prevent infinite waiting
        const timeout = setTimeout(() => {
          if (!user) {
            console.log('⚠️ Dashboard: User data timeout, proceeding with limited data');
            setLoading(false);
          }
        }, 5000); // 5 second timeout
        
        return () => clearTimeout(timeout);
      }
    } else if (!authLoading && !isAuthenticated) {
      console.log('🏠 Dashboard: Not authenticated');
      setLoading(false);
    } else {
      console.log('🏠 Dashboard: Waiting for auth...', { authLoading, isAuthenticated, hasUser: !!user });
    }
  }, [authLoading, isAuthenticated, user]);

  const loadDashboardData = async () => {
    try {
      console.log('📡 Loading dashboard data...');
      // Load user profile data which includes stats
      const response = await userAPI.getProfileData();
      
      console.log('✅ Dashboard data received:', response.data);
      console.log('📊 Stats:', response.data.stats);
      console.log('🚀 Startups data:', response.data.startups);
      
      setStats(response.data.stats);
      // Sort applications by date (newest first) and store all of them
      const allApplications = (response.data.applications || []).sort((a, b) => {
        const dateA = new Date(a.created_at || 0).getTime();
        const dateB = new Date(b.created_at || 0).getTime();
        return dateB - dateA; // Descending order (newest first)
      });
      setRecentActivity(allApplications);

      // Save user's startups
      const startups = response.data.startups || [];
      console.log(`📋 Number of startups: ${startups.length}`);
      
      // Log each startup's details
      startups.forEach((s, index) => {
        console.log(`\n🏢 Startup ${index + 1}:`);
        console.log(`  - ID: ${s.id}`);
        console.log(`  - Title: ${s.title}`);
        console.log(`  - Type: ${s.type}`);
        console.log(`  - Category: ${s.category}`);
        console.log(`  - Description: ${s.description?.substring(0, 50)}...`);
        console.log(`  - Revenue: ${s.revenue}`);
        console.log(`  - Profit: ${s.profit}`);
        console.log(`  - Asking Price: ${s.asking_price}`);
        console.log(`  - Earn Through: ${s.earn_through}`);
        console.log(`  - Team Size: ${s.team_size}`);
      });
      
      setMyStartups(startups);

      // Fetch positions for each startup (inline listing)
      if (startups.length > 0) {
        const positionPromises = startups.map(async (s) => {
          try {
            const res = await positionAPI.getStartupPositions(s.id);
            const positions = res.data?.positions || (Array.isArray(res.data) ? res.data : []);
            return [s.id, positions];
          } catch (e) {
            return [s.id, []];
          }
        });
        const results = await Promise.all(positionPromises);
        const map = results.reduce((acc, [id, positions]) => {
          acc[id] = positions;
          return acc;
        }, {});
        setStartupIdToPositions(map);
      } else {
        setStartupIdToPositions({});
      }
    } catch (error) {
      console.error('Failed to load dashboard data:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteStartup = async (startupId, startupTitle) => {
    console.log('🗑️ Delete startup initiated:', { startupId, startupTitle });
    
    const confirmed = window.confirm(
      `Are you sure you want to delete "${startupTitle}"?\n\nThis action cannot be undone and will remove all associated positions and data.`
    );
    
    if (!confirmed) {
      console.log('❌ Delete cancelled by user');
      return;
    }

    console.log('✅ User confirmed deletion, proceeding...');

    try {
      console.log('📡 Sending DELETE request to:', `/api/startups/${startupId}`);
      
      const response = await apiClient.delete(`/api/startups/${startupId}`);
      
      console.log('✅ DELETE request successful:', response);

      // Remove the startup from the local state
      setMyStartups(prev => {
        const updated = prev.filter(s => s.id !== startupId);
        console.log('📊 Updated startups list. Remaining:', updated.length);
        return updated;
      });
      
      // Also remove its positions from the map
      setStartupIdToPositions(prev => {
        const updated = { ...prev };
        delete updated[startupId];
        console.log('📊 Updated positions map');
        return updated;
      });
      
      alert('Startup deleted successfully!');
      console.log('✅ Startup deletion completed successfully');
    } catch (error) {
      console.error('❌ Error deleting startup:', error);
      console.error('Error details:', {
        message: error.message,
        response: error.response?.data,
        status: error.response?.status,
        statusText: error.response?.statusText
      });
      
      const errorMessage = error.response?.data?.error || error.response?.data?.message || 'Unknown error';
      alert(`Failed to delete startup: ${errorMessage}`);
    }
  };

  if (authLoading || loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-xl">
          {authLoading ? 'Authenticating...' : 'Loading dashboard...'}
        </div>
      </div>
    );
  }

  const renderEntrepreneurDashboard = () => (
    <div className={styles.dashboard}>
      <div className={styles.welcome}>
        <h1>Welcome, {user?.username}!</h1>
        <p>Manage your startups and find team members</p>
      </div>

      <div className={styles.statsGrid}>
        <div className={styles.statCard}>
          <h3>My Startups</h3>
          <div className={styles.statNumber}>{stats?.startups_created || 0}</div>
          <Link to="/createstartup" className={styles.statAction}>
            Create New Startup
          </Link>
        </div>

        <div className={styles.statCard}>
          <h3>Applications Received</h3>
          <div className={styles.statNumber}>{stats?.applications_received || 0}</div>
          <Link to="/startups-with-applications" className={styles.statAction}>
            View Applications
          </Link>
        </div>

        <div className={styles.statCard}>
          <h3>Profile Views</h3>
          <div className={styles.statNumber}>0</div>
          <Link to="/account" className={styles.statAction}>
            Update Profile
          </Link>
        </div>
      </div>

      <div className={styles.quickActions}>
        <h2>Quick Actions</h2>
        <div className={styles.actionGrid}>
          <Link to="/createstartup" className={styles.actionCard}>
            <h3>Create Startup Listing</h3>
            <p>List your startup for funding or collaboration</p>
          </Link>
          
          <Link to="/search" className={styles.actionCard}>
            <h3>Search Startups</h3>
            <p>Find startups to collaborate with or invest in</p>
          </Link>
          
          <Link to="/message" className={styles.actionCard}>
            <h3>Messages</h3>
            <p>Connect with potential team members</p>
          </Link>
        </div>
      </div>

      {/* My Startups Section */}
      <div className={styles.section}>
        <h2>My Startups</h2>
        {myStartups.length === 0 ? (
          <div className={styles.emptyState}>
            <p>You have no startups yet. Create one to get started.</p>
            <Link to="/createstartup" className={styles.statAction}>Create Startup</Link>
          </div>
        ) : (
          <div className={styles.cardsGrid}>
            {myStartups.map((startup) => (
              <div key={startup.id} className={styles.startupCard}>
                <div className={styles.cardHeader}>
                  <div className={styles.cardIcon}>
                    <h3>{startup.title}</h3>
                  </div>
                  <span className={styles.cardTag}>
                    {startup.category ? startup.category.charAt(0).toUpperCase() + startup.category.slice(1) : 'Other'}
                  </span>
                </div>
                <p className={styles.cardDescription}>
                  {startup.description?.substring(0, 100)}
                  {startup.description && startup.description.length > 100 ? '...' : ''}
                </p>
                
                {/* Conditional stats based on startup type */}
                {startup.type === 'marketplace' ? (
                  <div className={styles.cardStats}>
                    <div>
                      <span className={styles.statsLabel}>Revenue</span>
                      <p className={styles.statsValue}>{startup.revenue || '$0'}</p>
                    </div>
                    <div>
                      <span className={styles.statsLabel}>Profit</span>
                      <p className={styles.statsValue}>{startup.profit || '$0'}</p>
                    </div>
                    <div>
                      <span className={styles.statsLabel}>Price</span>
                      <p className={styles.statsValue}>{startup.asking_price || '$0'}</p>
                    </div>
                  </div>
                ) : (
                  <div className={styles.cardStats}>
                    <div>
                      <span className={styles.statsLabel}>Earn Through</span>
                      <p className={styles.statsValue}>{startup.earn_through || 'N/A'}</p>
                    </div>
                    <div>
                      <span className={styles.statsLabel}>Category</span>
                      <p className={styles.statsValue}>{startup.category ? startup.category.charAt(0).toUpperCase() + startup.category.slice(1) : 'N/A'}</p>
                    </div>
                    <div>
                      <span className={styles.statsLabel}>Team Size</span>
                      <p className={styles.statsValue}>{startup.team_size || 'N/A'}</p>
                    </div>
                  </div>
                )}
                
                <div className={styles.cardActions}>
                  <Link to={`/startupdetail/${startup.id}`} className={styles.viewButton}>
                    View Details
                  </Link>
                  <button 
                    onClick={() => handleDeleteStartup(startup.id, startup.title)} 
                    className={styles.deleteButton}
                  >
                    Delete
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* My Positions inline */}
      <div className={styles.section}>
        <h2>My Positions</h2>
        {myStartups.length === 0 ? (
          <div className={styles.emptyState}>
            <p>You have no startups yet. Create one to start hiring.</p>
            <Link to="/createstartup" className={styles.statAction}>Create Startup</Link>
          </div>
        ) : myStartups.filter((s) => s.type === 'collaboration').length === 0 ? (
          <div className={styles.emptyState}>
            <p>You have no collaboration startups. Positions can only be managed for collaboration-type startups.</p>
          </div>
        ) : (
          <div className={styles.listGroup}>
            {myStartups
              .filter((s) => s.type === 'collaboration')
              .map((s) => {
              const positions = startupIdToPositions[s.id] || [];
              const openCount = positions.filter(p => p.is_active).length;
              return (
                <div key={s.id} className={styles.itemCard}>
                  <div className={styles.itemHeader}>
                    <div>
                      <h3>{s.title}</h3>
                      {/*<p>{s.description?.substring(0, 120)}{(s.description && s.description.length > 120) ? '…' : ''}</p>*/}
                    </div>
                    {s.type === 'collaboration' && (
                      <div className={styles.itemActions}>
                        <Link to={`/startups/${s.id}/positions`} className={styles.statAction}>Manage Positions</Link>
                      </div>
                    )}
                  </div>
                  {s.type === 'collaboration' && (
                    positions.length === 0 ? (
                      <div className={styles.emptyRow}>
                        <span>No positions yet.</span>
                        <Link to={`/startups/${s.id}/positions`} className={styles.statAction}>Create Position</Link>
                      </div>
                    ) : (
                      <div className={styles.table}>
                        <div className={styles.tableHeader}>
                          <div>Title</div>
                          <div>Status</div>
                          <div>Applications</div>
                          <div>Posted</div>
                          <div></div>
                        </div>
                        {positions.map((p) => (
                          <div key={p.id} className={styles.tableRow}>
                            <div>{p.title}</div>
                            <div>
                              <span className={p.is_active ? styles.statusActive : styles.statusClosed}>
                                {p.is_active ? 'Active' : 'Closed'}
                              </span>
                            </div>
                            <div>{p.applications_count || 0}</div>
                            <div>{new Date(p.created_at).toLocaleDateString()}</div>
                            <div>
                              <Link to={`/positions/${p.id}/applications`} className={styles.statAction}>
                                View Applications
                              </Link>
                            </div>
                          </div>
                        ))}
                      </div>
                    )
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );

  const renderStudentDashboard = () => (
    <div className={styles.dashboard}>
      <div className={styles.welcome}>
        <h1>Welcome, {user?.username}!</h1>
        <p>Find opportunities and join exciting startups</p>
      </div>

      <div className={styles.statsGrid}>
        <div className={styles.statCard}>
          <h3>Applications Sent</h3>
          <div className={styles.statNumber}>{stats?.applications_submitted || 0}</div>
          <button 
            onClick={scrollToRecentApplications}
            className={styles.statAction}
          >
            View Applications
          </button>
        </div>

        <div className={styles.statCard}>
          <h3>Favorites</h3>
          <div className={styles.statNumber}>{stats?.favorites_count || 0}</div>
          <Link to="/account" className={styles.statAction}>
            View Favorites
          </Link>
        </div>

        <div className={styles.statCard}>
          <h3>Profile Views</h3>
          <div className={styles.statNumber}>0</div>
          <Link to="/account" className={styles.statAction}>
            Update Profile
          </Link>
        </div>
      </div>

      <div className={styles.quickActions}>
        <h2>Quick Actions</h2>
        <div className={styles.actionGrid}>
          <Link to="/search" className={styles.actionCard}>
            <h3>Search Opportunities</h3>
            <p>Find startups and positions that match your skills</p>
          </Link>
          
          <Link to="/collaboration" className={styles.actionCard}>
            <h3>Browse Collaborations</h3>
            <p>Discover startups looking for team members</p>
          </Link>
          
          <Link to="/message" className={styles.actionCard}>
            <h3>Messages</h3>
            <p>Connect with startup founders</p>
          </Link>
        </div>
      </div>

      <div className={styles.section} ref={recentApplicationsRef}>
        <div className={styles.sectionHeader}>
          <h2>Recent Applications</h2>
          <p>Track the latest updates on your collaboration submissions</p>
        </div>

        {recentActivity.length === 0 ? (
          <div className={styles.emptyState}>
            <h3>No applications yet</h3>
            <p>Ready to get started? Explore collaborations and apply to your favorite startups.</p>
            <Link to="/collaboration" className={styles.statAction}>
              Find Collaborations
            </Link>
          </div>
        ) : (
          <>
            <div className={styles.tableWrapper}>
              <div className={styles.tableHeader}>
                <div>Startup</div>
                <div>Position</div>
                <div>Status</div>
                <div>Submitted</div>
                <div></div>
              </div>
              {displayedApplications.map((application) => (
                <div key={application.id} className={styles.tableRow}>
                  <div>{application.startup?.title || 'Unknown Startup'}</div>
                  <div>{application.position?.title || 'General Collaboration'}</div>
                  <div>
                    <span className={getStatusBadgeClass(application.status)}>
                      {formatStatusLabel(application.status)}
                    </span>
                  </div>
                  <div>{formatDate(application.created_at)}</div>
                  <div>
                    {application.startup?.id && (
                      <Link to={`/startupdetail/${application.startup.id}`} className={styles.statAction}>
                        View Startup
                      </Link>
                    )}
                  </div>
                </div>
              ))}
            </div>
            
            {totalPages > 1 && (
              <div className={styles.pagination}>
                <button
                  onClick={handlePreviousPage}
                  disabled={currentPage === 1}
                  className={styles.paginationButton}
                >
                  ←
                </button>
                <span className={styles.paginationInfo}>
                  Page {currentPage} of {totalPages}
                </span>
                <button
                  onClick={handleNextPage}
                  disabled={currentPage === totalPages}
                  className={styles.paginationButton}
                >
                  →
                </button>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );

  const renderInvestorDashboard = () => (
    <div className={styles.dashboard}>
      <div className={styles.welcome}>
        <h1>Welcome, {user?.username}!</h1>
        <p>Discover and invest in promising startups</p>
      </div>

      <div className={styles.statsGrid}>
        <div className={styles.statCard}>
          <h3>Favorites</h3>
          <div className={styles.statNumber}>{stats?.favorites_count || 0}</div>
          <Link to="/investor-dashboard?tab=favorites" className={styles.statAction}>
            View Favorites
          </Link>
        </div>

        <div className={styles.statCard}>
          <h3>Interests Expressed</h3>
          <div className={styles.statNumber}>{stats?.interests_count || 0}</div>
          <Link to="/investor-dashboard?tab=interests" className={styles.statAction}>
            View Interests
          </Link>
        </div>

        <div className={styles.statCard}>
          <h3>Profile Views</h3>
          <div className={styles.statNumber}>0</div>
          <Link to="/account" className={styles.statAction}>
            Update Profile
          </Link>
        </div>
      </div>

      <div className={styles.quickActions}>
        <h2>Quick Actions</h2>
        <div className={styles.actionGrid}>
          <Link to="/investor-dashboard" className={styles.actionCard}>
            <h3>Investor Dashboard</h3>
            <p>Manage your investments and interests</p>
          </Link>
          
          <Link to="/search" className={styles.actionCard}>
            <h3>Find Investments</h3>
            <p>Search for promising investment opportunities</p>
          </Link>
          
          <Link to="/message" className={styles.actionCard}>
            <h3>Messages</h3>
            <p>Connect with entrepreneurs</p>
          </Link>
        </div>
      </div>
    </div>
  );

  // If we have authentication but no user data, show a basic dashboard
  const renderFallbackDashboard = () => (
    <div className={styles.dashboard}>
      <div className={styles.welcome}>
        <h1>Welcome to your Dashboard!</h1>
        <p>Loading your profile data...</p>
      </div>
      <div className={styles.quickActions}>
        <h2>Quick Actions</h2>
        <div className={styles.actionGrid}>
          <Link to="/search" className={styles.actionCard}>
            <h3>Search</h3>
            <p>Find opportunities and startups</p>
          </Link>
          <Link to="/account" className={styles.actionCard}>
            <h3>Profile</h3>
            <p>Update your profile information</p>
          </Link>
          <Link to="/message" className={styles.actionCard}>
            <h3>Messages</h3>
            <p>Connect with others</p>
          </Link>
        </div>
      </div>
    </div>
  );

  return (
    <>
      <Navbar />
      <div className={styles.container}>
        {user ? (
          <>
            {isEntrepreneur() && renderEntrepreneurDashboard()}
            {isStudent() && renderStudentDashboard()}
            {isInvestor() && renderInvestorDashboard()}
          </>
        ) : (
          renderFallbackDashboard()
        )}
      </div>
      <Footer />
    </>
  );
};

export default Dashboard;
