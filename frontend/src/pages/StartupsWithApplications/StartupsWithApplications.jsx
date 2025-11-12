import React, { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Navbar } from '../../components/Navbar/Navbar';
import { Footer } from '../../components/Footer/Footer';
import styles from './StartupsWithApplications.module.css';
import { userAPI } from '../../utils/apiServices';
import { useAuth } from '../../contexts/AuthContext';
import { toast } from 'react-toastify';

const StartupsWithApplications = () => {
  const navigate = useNavigate();
  const { user } = useAuth();
  const [startups, setStartups] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadStartups();
  }, []);

  const loadStartups = async () => {
    try {
      setLoading(true);
      const response = await userAPI.getUserStartupsWithApplications();
      console.log('Startups with applications response:', response.data);
      
      const results = response.data.results || [];
      
      // Deduplicate startups by ID (safety measure)
      const uniqueStartups = results.reduce((acc, startup) => {
        if (!acc.find(s => s.id === startup.id)) {
          acc.push(startup);
        }
        return acc;
      }, []);
      
      console.log(`Loaded ${uniqueStartups.length} unique startups (from ${results.length} total)`);
      setStartups(uniqueStartups);
    } catch (error) {
      console.error('Failed to load startups:', error);
      toast.error('Failed to load startups with applications');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <>
        <Navbar />
        <div className={styles.container}>
          <div className={styles.loading}>Loading startups with applications...</div>
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
            <Link to="/dashboard" className={styles.backLink}>
              ← Back to Dashboard
            </Link>
            <h1>Startups with Applications</h1>
            <p>View and manage applications received by your startups</p>
          </div>
        </div>

        <div className={styles.startupsSection}>
          {startups.length === 0 ? (
            <div className={styles.emptyState}>
              <h4>No applications received yet</h4>
              <p>Your startups haven't received any applications yet. Share your positions to attract candidates.</p>
            </div>
          ) : (
            <div className={styles.startupsList}>
              {startups.map((startup) => (
                <div key={startup.id} className={styles.startupCard}>
                  <div className={styles.cardHeader}>
                    <div className={styles.startupInfo}>
                      <h3>{startup.title}</h3>
                      <p className={styles.startupDescription}>
                        {startup.description?.substring(0, 150)}...
                      </p>
                      <div className={styles.startupMeta}>
                        <span className={styles.categoryTag}>{startup.category || 'Other'}</span>
                        <span className={styles.typeTag}>{startup.type || 'N/A'}</span>
                      </div>
                    </div>
                    <div className={styles.applicationCount}>
                      <div className={styles.countNumber}>{startup.applications_count || 0}</div>
                      <div className={styles.countLabel}>
                        Application{startup.applications_count !== 1 ? 's' : ''}
                      </div>
                    </div>
                  </div>

                  <div className={styles.cardActions}>
                    <Link
                      to={`/startups/${startup.id}/applications`}
                      className={styles.viewButton}
                      onClick={(e) => {
                        if (!startup.id) {
                          e.preventDefault();
                          toast.error('Invalid startup ID');
                          console.error('Startup ID is missing:', startup);
                        }
                      }}
                    >
                      View Applications
                    </Link>
                    <Link
                      to={`/startupdetail/${startup.id}`}
                      className={styles.detailsButton}
                    >
                      View Startup Details
                    </Link>
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

export default StartupsWithApplications;

