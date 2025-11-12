import React, { useState, useEffect } from 'react';
import { useParams, Link, useNavigate } from 'react-router-dom';
import { Navbar } from '../../components/Navbar/Navbar';
import { Footer } from '../../components/Footer/Footer';
import styles from './PositionApplications.module.css';
import { positionAPI, applicationAPI } from '../../utils/apiServices';
import { useAuth } from '../../contexts/AuthContext';
import { toast } from 'react-toastify';

const PositionApplications = () => {
  const { positionId } = useParams();
  const navigate = useNavigate();
  const { user } = useAuth();
  const [applications, setApplications] = useState([]);
  const [position, setPosition] = useState(null);
  const [startup, setStartup] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadApplications();
  }, [positionId]);

  const loadApplications = async () => {
    try {
      setLoading(true);
      const response = await positionAPI.getPositionApplications(positionId);
      console.log('Position applications response:', response.data);
      
      setApplications(response.data.applications || []);
      setPosition(response.data.position);
      setStartup(response.data.startup);
    } catch (error) {
      console.error('Failed to load applications:', error);
      toast.error('Failed to load applications');
      navigate('/dashboard');
    } finally {
      setLoading(false);
    }
  };

  const handleApproveApplication = async (applicationId) => {
    console.log('Approve button clicked for application:', applicationId);
    const confirmed = window.confirm('Are you sure you want to approve this application?');
    console.log('User confirmed:', confirmed);
    if (!confirmed) return;
    
    try {
      console.log('Sending approve request for application:', applicationId);
      const response = await applicationAPI.approveApplication(applicationId);
      console.log('Approve response:', response);
      toast.success('Application approved successfully! Opening chat...');
      
      // Navigate to messaging page with conversation ID
      const conversationId = response.data?.conversation_id;
      if (conversationId) {
        setTimeout(() => {
          navigate(`/message?conversation=${conversationId}`);
        }, 1000);
      } else {
        // Fallback: navigate to messages page
        setTimeout(() => {
          navigate('/message');
        }, 1000);
      }
      
      loadApplications();
    } catch (error) {
      console.error('Failed to approve application:', error);
      console.error('Error response:', error.response);
      toast.error(error.response?.data?.detail || 'Failed to approve application');
    }
  };

  const handleDeclineApplication = async (applicationId) => {
    console.log('Decline button clicked for application:', applicationId);
    const confirmed = window.confirm('Are you sure you want to decline this application?');
    console.log('User confirmed:', confirmed);
    if (!confirmed) return;
    
    try {
      console.log('Sending decline request for application:', applicationId);
      const response = await applicationAPI.declineApplication(applicationId);
      console.log('Decline response:', response);
      toast.success('Application declined');
      // Reload applications to get updated status
      await loadApplications();
    } catch (error) {
      console.error('Failed to decline application:', error);
      console.error('Error response:', error.response);
      toast.error(error.response?.data?.detail || 'Failed to decline application');
    }
  };

  const getStatusBadgeClass = (status) => {
    switch (status) {
      case 'approved':
        return styles.statusApproved;
      case 'rejected':
        return styles.statusRejected;
      case 'pending':
        return styles.statusPending;
      default:
        return styles.statusPending;
    }
  };

  const formatDate = (dateString) => {
    if (!dateString) return '-';
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'long',
      day: 'numeric'
    });
  };

  if (loading) {
    return (
      <>
        <Navbar />
        <div className={styles.container}>
          <div className={styles.loading}>Loading applications...</div>
        </div>
        <Footer />
      </>
    );
  }

  return (
    <>
      <Navbar />
      <div className={styles.container}>
        <div className={styles.header}>
          <div className={styles.headerContent}>
            <div>
              <Link to={`/startups/${startup?.id}/positions`} className={styles.backLink}>
                ← Back to Positions
              </Link>
              <h1>Applications for Position</h1>
              {position && (
                <div className={styles.positionInfo}>
                  <h2>{position.title}</h2>
                  {startup && (
                    <p className={styles.startupName}>at {startup.title}</p>
                  )}
                  {position.description && (
                    <p className={styles.positionDescription}>{position.description}</p>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>

        <div className={styles.applicationsSection}>
          <div className={styles.sectionHeader}>
            <h3>Applications ({applications.length})</h3>
            <p>Review and manage applications for this position</p>
          </div>

          {applications.length === 0 ? (
            <div className={styles.emptyState}>
              <h4>No applications yet</h4>
              <p>No one has applied for this position yet. Check back later or share the position to attract candidates.</p>
            </div>
          ) : (
            <div className={styles.applicationsList}>
              {applications.map((application) => (
                <div key={application.id} className={styles.applicationCard}>
                  <div className={styles.applicationHeader}>
                    <div className={styles.applicantInfo}>
                      <h4>{application.applicant?.username || 'Unknown Applicant'}</h4>
                      <p className={styles.applicantEmail}>{application.applicant?.email || ''}</p>
                      <p className={styles.applicationDate}>
                        Applied on {formatDate(application.created_at)}
                      </p>
                    </div>
                    <span className={`${styles.statusBadge} ${getStatusBadgeClass(application.status)}`}>
                      {application.status.charAt(0).toUpperCase() + application.status.slice(1)}
                    </span>
                  </div>

                  <div className={styles.applicationContent}>
                    {application.cover_letter && (
                      <div className={styles.contentSection}>
                        <h5>Cover Letter</h5>
                        <p>{application.cover_letter}</p>
                      </div>
                    )}

                    {application.experience && (
                      <div className={styles.contentSection}>
                        <h5>Experience</h5>
                        <p>{application.experience}</p>
                      </div>
                    )}

                    {application.portfolio_url && (
                      <div className={styles.contentSection}>
                        <h5>Portfolio</h5>
                        <a 
                          href={application.portfolio_url} 
                          target="_blank" 
                          rel="noopener noreferrer"
                          className={styles.portfolioLink}
                        >
                          {application.portfolio_url}
                        </a>
                      </div>
                    )}

                    {application.resume_url && (
                      <div className={styles.contentSection}>
                        <h5>Resume/CV</h5>
                        <a 
                          href={application.resume_url} 
                          target="_blank" 
                          rel="noopener noreferrer"
                          download
                          className={styles.resumeLink}
                        >
                          📄 Download CV
                        </a>
                      </div>
                    )}
                  </div>

                  {application.status === 'pending' && (
                    <div className={styles.applicationActions}>
                      <button
                        type="button"
                        onClick={(e) => {
                          e.preventDefault();
                          e.stopPropagation();
                          handleApproveApplication(application.id);
                        }}
                        className={styles.approveButton}
                      >
                        Approve
                      </button>
                      <button
                        type="button"
                        onClick={(e) => {
                          e.preventDefault();
                          e.stopPropagation();
                          handleDeclineApplication(application.id);
                        }}
                        className={styles.declineButton}
                      >
                        Decline
                      </button>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
      <Footer />
    </>
  );
};

export default PositionApplications;

