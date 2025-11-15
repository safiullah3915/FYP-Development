import React, { useState } from "react";
import styles from "./CollaborationCard.module.css";
import { Link } from 'react-router-dom';
import { getStartupDetailPath, normalizeId } from '../../utils/idUtils';

const CollaborationCard = ({ 
  startup, 
  score, 
  matchReasons, 
  appliedPositionIds, 
  appliedStartupIds,
  // Legacy props for backward compatibility
  id, 
  title, 
  description, 
  earn_through, 
  phase, 
  team_size, 
  category, 
  type 
}) => {
  const [showPositions, setShowPositions] = useState(false);
  
  // Use startup object if provided, otherwise use legacy props
  const startupData = startup || { id, title, description, earn_through, phase, team_size, category, type };
  const startupId = startupData.id;
  const positions = startupData.positions || startupData.open_positions || [];
  const normalizedStartupId = normalizeId(startupId);
  const primaryPosition = startupData.primary_position || startupData.top_position || positions[0];
  
  if (!positions || positions.length === 0) {
    return null;
  }
  
  // Calculate match percentage from score
  const matchPercentage = score ? Math.round(score * 100) : null;
  
  // Always navigate to startup detail page
  const getLinkDestination = () => {
      return getStartupDetailPath(normalizedStartupId);
  };

  return (
    <div className={styles.cardWrapper}>
      <Link to={getLinkDestination()} className={styles.linkWrapper}>
        <div className={styles.card}>
          <div className={styles.cardHeader}>
            <div className={styles.icon}>
              <img src="/diamond.svg" alt="" />
              <img src="/Decentralized Network.svg" alt="" />
              <h3>{startupData.title || 'Startup Name'}</h3>
            </div>
            <div className={styles.headerRight}>
              <span className={styles.tag}>{startupData.category || 'Collaboration'}</span>
              {matchPercentage && (
                <span className={styles.matchBadge}>{matchPercentage}% Match</span>
              )}
            </div>
          </div>
      
      {primaryPosition && (
        <div className={styles.primaryPosition}>
          <div className={styles.primaryPositionInfo}>
            <span className={styles.positionLabel}>Top recommended role</span>
            <h4>{primaryPosition.title || 'Open role'}</h4>
            <p>{primaryPosition.description || primaryPosition.requirements || 'Help this startup grow by joining the core team.'}</p>
          </div>
          {primaryPosition.id && (
            <Link 
              to={`/apply-for-collaboration/${normalizedStartupId}?position=${primaryPosition.id}`}
              className={styles.primaryPositionCta}
              onClick={(e) => e.stopPropagation()}
            >
              Apply now
            </Link>
          )}
        </div>
      )}
          
          {/* Match Reasons */}
          {matchReasons && matchReasons.length > 0 && (
            <div className={styles.matchReasons}>
              <strong>Why this matches:</strong>
              <ul>
                {matchReasons.slice(0, 2).map((reason, idx) => (
                  <li key={idx}>{reason}</li>
                ))}
              </ul>
            </div>
          )}
          
          <p className={styles.description}>{startupData.description || 'No description available'}</p>
          <div className={styles.stats}>
            <div className={styles.statsheading}>
              Earn Through
              <p>{startupData.earn_through || 'Equity'}</p>
            </div>
            <div className={styles.statsheading}>
              Phase
              <p>{startupData.phase || 'N/A'}</p>
            </div>
            <div className={styles.statsheading}>
              Team Size
              <p>{startupData.team_size || '1-5'}</p>
            </div>
          </div>
        </div>
      </Link>
      
      {/* Positions Section */}
      <div className={styles.positionsSection}>
        <button 
          className={styles.positionsToggle}
          onClick={() => setShowPositions(!showPositions)}
        >
          {showPositions ? '▼' : '▶'} View all {positions.length} open position{positions.length !== 1 ? 's' : ''}
        </button>
        
        {showPositions && (
          <div className={styles.positionsList}>
            {positions.map((position) => {
              const hasApplied = appliedPositionIds?.has(position.id) || appliedStartupIds?.has(startupId);
              return (
                <div key={position.id} className={styles.positionItem}>
                  <div className={styles.positionHeader}>
                    <strong>{position.title}</strong>
                    {hasApplied && <span className={styles.appliedBadge}>Applied</span>}
                  </div>
                  <p className={styles.positionDesc}>{position.description || position.requirements}</p>
                  {!hasApplied && (
                    <Link 
                      to={`/apply-for-collaboration/${normalizedStartupId}?position=${position.id}`}
                      className={styles.applyButton}
                      onClick={(e) => e.stopPropagation()}
                    >
                      Apply
                    </Link>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
};

export default CollaborationCard;
