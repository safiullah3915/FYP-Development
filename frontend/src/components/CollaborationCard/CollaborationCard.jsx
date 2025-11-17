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
  const normalizedStartupId = normalizeId(startupId);
  
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
      
      {/* primary position removed to keep consistent card size */}
          
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
      
      {/* positions removed to keep card layout consistent */}
    </div>
  );
};

export default CollaborationCard;
