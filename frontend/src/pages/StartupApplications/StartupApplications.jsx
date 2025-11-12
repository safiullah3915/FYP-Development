import React, { useState, useEffect } from 'react';
import { useParams, Link, useNavigate } from 'react-router-dom';
import { Navbar } from '../../components/Navbar/Navbar';
import { Footer } from '../../components/Footer/Footer';
import styles from './StartupApplications.module.css';
import { startupAPI, applicationAPI } from '../../utils/apiServices';
import { toast } from 'react-toastify';

const StartupApplications = () => {
  const { startupId } = useParams();
  const navigate = useNavigate();
  const [applications, setApplications] = useState([]);
  const [startup, setStartup] = useState(null);
  const [loading, setLoading] = useState(true);
  const [groupedByPosition, setGroupedByPosition] = useState({});

  useEffect(() => {
    loadApplications();
  }, [startupId]);

  const loadApplications = async () => {
    try {
      setLoading(true);
      const response = await startupAPI.getStartupApplications(startupId);
      console.log('Startup applications response:', response.data);
      
      const apps = response.data.applications || [];
      setApplications(apps);
      setStartup(response.data.startup);
      
      // Group applications by position
      const grouped = apps.reduce((acc, app) => {
        const positionId = app.position?.id;
        if (positionId) {
          if (!acc[positionId]) {
            acc[positionId] = {
              position: app.position,
              applications: []
            };
          }
          acc[positionId].applications.push(app);
        }
        return acc;
      }, {});
      
      setGroupedByPosition(grouped);
    } catch (error) {
      console.error('Failed to load applications:', error);
      console.error('Error details:', error.response?.data || error.message);
      toast.error(error.response?.data?.detail || 'Failed to load applications');
      // Don't redirect on error, just show error message
      // navigate('/dashboard');
    } finally {
      setLoading(false);
    }
  };

  const handleApproveApplication = async (applicationId) => {
    const confirmed = window.confirm('Are you sure you want to approve this application?');
    if (!confirmed) return;

    try {
      await applicationAPI.approveApplication(applicationId);
      toast.success('Application approved successfully');
      loadApplications();
    } catch (error) {
      console.error('Failed to approve application:', error);
      toast.error('Failed to approve application');
    }
  };

  const handleDeclineApplication = async (applicationId) => {
    const confirmed = window.confirm('Are you sure you want to decline this application?');
    if (!confirmed) return;
    
    try {
      await applicationAPI.declineApplication(applicationId);
      toast.success('Application declined');
      loadApplications();
    } catch (error) {
      console.error('Failed to decline application:', error);
      toast.error('Failed to decline application');
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
            <Link to="/startups-with-applications" className={styles.backLink}>
              ← Back to Startups
            </Link>
            <h1>Applications for Startup</h1>
            {startup && (
              <div className={styles.startupInfo}>
                <h2>{startup.title}</h2>
                <p className={styles.startupDescription}>{startup.description}</p>
              </div>
            )}
          </div>
        </div>

        <div className={styles.applicationsSection}>
          <div className={styles.sectionHeader}>
            <h3>All Applications ({applications.length})</h3>
            <p>Applications grouped by position</p>
          </div>

          {applications.length === 0 ? (
            <div className={styles.emptyState}>
              <h4>No applications yet</h4>
              <p>This startup hasn't received any applications yet.</p>
            </div>
          ) : (
            <div className={styles.positionsContainer}>
              {Object.entries(groupedByPosition).map(([positionId, group]) => (
                <div key={positionId} className={styles.positionGroup}>
                  <div className={styles.positionHeader}>
                    <div>
                      <h4 className={styles.positionTitle}>{group.position?.title || 'Unknown Position'}</h4>
                      <Link
                        to={`/positions/${positionId}/applications`}
                        className={styles.viewPositionLink}
                      >
                        View Position Applications ({group.applications.length})
                      </Link>
                    </div>
                  </div>

                  <div className={styles.applicationsList}>
                    {group.applications.map((application) => (
                      <div key={application.id} className={styles.applicationCard}>
                        <div className={styles.applicationHeader}>
                          <div className={styles.applicantInfo}>
                            <h5>{application.applicant?.username || 'Unknown Applicant'}</h5>
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
                              <h6>Cover Letter</h6>
                              <p>{application.cover_letter.length > 200 ? application.cover_letter.substring(0, 200) + '...' : application.cover_letter}</p>
                            </div>
                          )}

                          {application.experience && (
                            <div className={styles.contentSection}>
                              <h6>Experience</h6>
                              <p>{application.experience.length > 200 ? application.experience.substring(0, 200) + '...' : application.experience}</p>
                            </div>
                          )}

                          {application.portfolio_url && (
                            <div className={styles.contentSection}>
                              <h6>Portfolio</h6>
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
                              <h6>Resume/CV</h6>
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
                              onClick={() => handleApproveApplication(application.id)}
                              className={styles.approveButton}
                            >
                              Approve
                            </button>
                            <button
                              onClick={() => handleDeclineApplication(application.id)}
                              className={styles.declineButton}
                            >
                              Decline
                            </button>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
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

export default StartupApplications;

