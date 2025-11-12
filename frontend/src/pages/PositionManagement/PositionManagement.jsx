import React, { useState, useEffect } from 'react';
import { useParams, Link, useNavigate } from 'react-router-dom';
import { Navbar } from '../../components/Navbar/Navbar';
import { Footer } from '../../components/Footer/Footer';
import styles from './PositionManagement.module.css';
import { positionAPI, startupAPI } from '../../utils/apiServices';
import { useAuth } from '../../contexts/AuthContext';
import { toast } from 'react-toastify';

const PositionManagement = () => {
  const { startupId } = useParams();
  const { user } = useAuth();
  const navigate = useNavigate();
  const [positions, setPositions] = useState([]);
  const [startup, setStartup] = useState(null);
  const [loading, setLoading] = useState(true);
  const [deletingPosition, setDeletingPosition] = useState(false);
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [newPosition, setNewPosition] = useState({
    title: '',
    description: '',
    requirements: ''
  });

  useEffect(() => {
    loadPositions();
  }, [startupId]);

  const loadPositions = async () => {
    try {
      setLoading(true);
      const response = await positionAPI.getStartupPositions(startupId);
      console.log('Positions API response:', response.data);
      
      let startupData = null;
      if (response.data.positions) {
        // Response has the expected format with startup and positions
        setPositions(response.data.positions || []);
        startupData = response.data.startup;
        setStartup(startupData);
      } else if (Array.isArray(response.data)) {
        // Response is just an array of positions
        setPositions(response.data);
        // Load startup info separately if needed
        try {
          const startupResponse = await startupAPI.getStartup(startupId);
          startupData = startupResponse.data;
          setStartup(startupData);
        } catch (err) {
          console.error('Failed to load startup details:', err);
        }
      } else {
        setPositions([]);
      }
      
      // Check if startup type is marketplace - positions are not allowed
      if (startupData && startupData.type === 'marketplace') {
        toast.error('Positions can only be managed for Collaboration-type startups');
        navigate('/dashboard');
        return;
      }
    } catch (error) {
      console.error('Failed to load positions:', error);
      toast.error('Failed to load positions');
    } finally {
      setLoading(false);
    }
  };

  const handleCreatePosition = async (e) => {
    e.preventDefault();
    
    if (!newPosition.title.trim() || !newPosition.description.trim()) {
      toast.error('Please fill in all required fields');
      return;
    }

    try {
      await positionAPI.createPosition(startupId, newPosition);
      toast.success('Position created successfully!');
      setNewPosition({ title: '', description: '', requirements: '' });
      setShowCreateForm(false);
      loadPositions();
    } catch (error) {
      console.error('Failed to create position:', error);
      toast.error('Failed to create position');
    }
  };

  const handleTogglePosition = async (positionId, isActive, positionTitle) => {
    // Show confirmation dialog
    const confirmed = window.confirm(
      `Are you sure you want to ${isActive ? 'close' : 'reopen'} the position "${positionTitle}"?\n\n${isActive ? 'This will close the position and it will no longer be visible on the platform. You can still view applications for this position, but it cannot be reopened.' : 'This will make the position active again and visible on the platform.'}`
    );
    
    if (!confirmed) {
      return;
    }
    
    try {
      setDeletingPosition(true);
      
      if (isActive) {
        // Close position (sets is_active=False, but position remains in database)
        await positionAPI.closePosition(positionId);
        toast.success('Position closed successfully');
      } else {
        // Closed positions cannot be reopened per requirements
        toast.error('Closed positions cannot be reopened');
        return;
      }
      
      loadPositions();
    } catch (error) {
      console.error('Failed to toggle position:', error);
      toast.error('Failed to update position status');
    } finally {
      setDeletingPosition(false);
    }
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setNewPosition(prev => ({
      ...prev,
      [name]: value
    }));
  };

  if (loading) {
    return (
      <>
        <Navbar />
        <div className={styles.container}>
          <div className={styles.loading}>Loading positions...</div>
        </div>
        <Footer />
      </>
    );
  }

  return (
    <>
      <Navbar />
      {deletingPosition && (
        <div className={styles.progressOverlay}>
          <div className={styles.progressModal}>
            <div className={styles.progressSpinner}></div>
            <p className={styles.progressText}>Processing...</p>
          </div>
        </div>
      )}
      <div className={styles.container}>
        <div className={styles.header}>
          <div className={styles.headerContent}>
            <h1>Team Recruitment</h1>
            {startup && (
              <div className={styles.startupInfo}>
                <h2>{startup.title}</h2>
                <p>{startup.description.substring(0, 150)}...</p>
              </div>
            )}
          </div>
          <button
            onClick={() => setShowCreateForm(true)}
            className={styles.createButton}
          >
            + Create New Position
          </button>
        </div>

        {showCreateForm && (
          <div className={styles.modal}>
            <div className={styles.modalContent}>
              <div className={styles.modalHeader}>
                <h3>Create New Position</h3>
                <button
                  onClick={() => setShowCreateForm(false)}
                  className={styles.closeButton}
                >
                  ×
                </button>
              </div>
              
              <form onSubmit={handleCreatePosition} className={styles.form}>
                <div className={styles.formGroup}>
                  <label>Position Title *</label>
                  <input
                    type="text"
                    name="title"
                    value={newPosition.title}
                    onChange={handleInputChange}
                    placeholder="e.g., Frontend Developer, Marketing Manager"
                    required
                  />
                </div>

                <div className={styles.formGroup}>
                  <label>Job Description *</label>
                  <textarea
                    name="description"
                    value={newPosition.description}
                    onChange={handleInputChange}
                    placeholder="Describe the role, responsibilities, and what you're looking for..."
                    rows="4"
                    required
                  />
                </div>

                <div className={styles.formGroup}>
                  <label>Requirements & Qualifications</label>
                  <textarea
                    name="requirements"
                    value={newPosition.requirements}
                    onChange={handleInputChange}
                    placeholder="List required skills, experience, education, etc."
                    rows="3"
                  />
                </div>

                <div className={styles.formActions}>
                  <button
                    type="button"
                    onClick={() => setShowCreateForm(false)}
                    className={styles.cancelButton}
                  >
                    Cancel
                  </button>
                  <button type="submit" className={styles.submitButton}>
                    Create Position
                  </button>
                </div>
              </form>
            </div>
          </div>
        )}

        <div className={styles.positionsSection}>
          <div className={styles.sectionHeader}>
            <h3>All Positions</h3>
            <p>Manage your team recruitment posts and view applications</p>
            <div className={styles.positionStats}>
              <span>Active: {positions.filter(p => p.is_active).length}</span>
              <span>Closed: {positions.filter(p => !p.is_active).length}</span>
            </div>
          </div>

          {positions.length === 0 ? (
            <div className={styles.emptyState}>
              <h4>No positions created yet</h4>
              <p>Create your first team recruitment post to start building your team</p>
              <button
                onClick={() => setShowCreateForm(true)}
                className={styles.primaryButton}
              >
                Create First Position
              </button>
            </div>
          ) : (
            <>
              {/* Active Positions */}
              {positions.filter(p => p.is_active).length > 0 && (
                <div>
                  <h4 className={styles.subsectionTitle}>Active Positions</h4>
            <div className={styles.positionsGrid}>
                    {positions.filter(p => p.is_active).map((position) => (
                <div key={position.id} className={styles.positionCard}>
                  <div className={styles.cardHeader}>
                    <div className={styles.cardTitle}>
                      <h4>{position.title}</h4>
                            <span className={`${styles.badge} ${styles.active}`}>
                              Active
                      </span>
                    </div>
                  </div>

                  <div className={styles.cardContent}>
                    <p className={styles.description}>{position.description}</p>
                    
                    {position.requirements && (
                      <div className={styles.requirements}>
                        <strong>Requirements:</strong>
                        <p>{position.requirements}</p>
                      </div>
                    )}

                    <div className={styles.stats}>
                      <div className={styles.stat}>
                        <span className={styles.statNumber}>{position.applications_count || 0}</span>
                        <span className={styles.statLabel}>Applications</span>
                      </div>
                      <div className={styles.stat}>
                        <span className={styles.statNumber}>
                          {new Date(position.created_at).toLocaleDateString()}
                        </span>
                        <span className={styles.statLabel}>Posted</span>
                      </div>
                    </div>
                  </div>

                  <div className={styles.cardActions}>
                    <Link
                      to={`/positions/${position.id}/applications`}
                      className={styles.viewButton}
                    >
                      View Applications
                    </Link>
                    <button
                            onClick={() => handleTogglePosition(position.id, position.is_active, position.title)}
                            className={`${styles.toggleButton} ${styles.closeBtn}`}
                    >
                            Close
                    </button>
                  </div>
                </div>
              ))}
            </div>
                </div>
              )}

              {/* Closed Positions */}
              {positions.filter(p => !p.is_active).length > 0 && (
                <div style={{ marginTop: '3rem' }}>
                  <h4 className={styles.subsectionTitle}>Closed Positions</h4>
                  <div className={styles.positionsGrid}>
                    {positions.filter(p => !p.is_active).map((position) => (
                      <div key={position.id} className={styles.positionCard}>
                        <div className={styles.cardHeader}>
                          <div className={styles.cardTitle}>
                            <h4>{position.title}</h4>
                            <span className={`${styles.badge} ${styles.inactive}`}>
                              Closed
                            </span>
                          </div>
                        </div>

                        <div className={styles.cardContent}>
                          <p className={styles.description}>{position.description}</p>
                          
                          {position.requirements && (
                            <div className={styles.requirements}>
                              <strong>Requirements:</strong>
                              <p>{position.requirements}</p>
                            </div>
                          )}

                          <div className={styles.stats}>
                            <div className={styles.stat}>
                              <span className={styles.statNumber}>{position.applications_count || 0}</span>
                              <span className={styles.statLabel}>Applications</span>
                            </div>
                            <div className={styles.stat}>
                              <span className={styles.statNumber}>
                                {new Date(position.created_at).toLocaleDateString()}
                              </span>
                              <span className={styles.statLabel}>Posted</span>
                            </div>
                          </div>
                        </div>

                        <div className={styles.cardActions}>
                          <Link
                            to={`/positions/${position.id}/applications`}
                            className={styles.viewButton}
                          >
                            View Applications
                          </Link>
                          <button
                            disabled
                            className={`${styles.toggleButton} ${styles.disabledBtn}`}
                            title="Closed positions cannot be reopened"
                          >
                            Closed
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </>
          )}
        </div>

        <div className={styles.helpSection}>
          <h3>Tips for Better Recruitment</h3>
          <div className={styles.tips}>
            <div className={styles.tip}>
              <h4>Be Specific</h4>
              <p>Clearly define the role, responsibilities, and required skills to attract the right candidates.</p>
            </div>
            <div className={styles.tip}>
              <h4>Highlight Benefits</h4>
              <p>Mention equity, learning opportunities, flexible work arrangements, or other perks.</p>
            </div>
            <div className={styles.tip}>
              <h4>Set Expectations</h4>
              <p>Be clear about time commitment, remote/in-person work, and compensation structure.</p>
            </div>
          </div>
        </div>
      </div>
      <Footer />
    </>
  );
};

export default PositionManagement;