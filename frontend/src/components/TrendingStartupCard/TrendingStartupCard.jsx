import React from "react";
import { Link } from "react-router-dom";
import PropTypes from "prop-types";
import styles from "./TrendingStartupCard.module.css";
import { getStartupDetailPath } from "../../utils/idUtils";

const TrendingStartupCard = ({ startup }) => {
  // Verify startup object and ID
  console.log('🔍 [TrendingStartupCard] Received startup:', startup);
  console.log('🔍 [TrendingStartupCard] Startup ID:', startup?.id);
  
  if (!startup) {
    console.error('❌ [TrendingStartupCard] No startup data provided!');
    return (
      <div className={styles.card}>
        <p>Error: No startup data available</p>
      </div>
    );
  }
  
  if (!startup.id) {
    console.error('❌ [TrendingStartupCard] Startup missing ID!', startup);
    return (
      <div className={styles.card}>
        <p>Error: Startup ID is missing</p>
      </div>
    );
  }
  
  // Validate UUID format (basic check) - log for debugging but don't block
  const uuidRegex = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
  if (typeof startup.id !== 'string') {
    console.error('❌ [TrendingStartupCard] Startup ID is not a string!', typeof startup.id, startup.id);
    return (
      <div className={styles.card}>
        <p>Error: Invalid startup ID type</p>
      </div>
    );
  }
  
  // Log UUID validation for debugging but don't block rendering
  if (!uuidRegex.test(startup.id)) {
    console.warn('⚠️ [TrendingStartupCard] Startup ID does not match UUID format (but continuing):', startup.id);
  }
  
  const {
    id,
    title,
    description,
    category,
    type,
    trending_score = 0,
    popularity_score = 0,
    velocity_score = 0,
    view_count_24h = 0,
    view_count_7d = 0,
    application_count_7d = 0,
    favorite_count_7d = 0,
    active_positions_count = 0,
  } = startup;

  // Calculate trending intensity (0-100)
  const trendingIntensity = Math.min(Math.round(trending_score * 100), 100);
  const popularityIntensity = Math.min(Math.round(popularity_score * 100), 100);

  // Determine trending badge color based on score
  const getTrendingBadgeClass = () => {
    if (trendingIntensity >= 70) return styles.trendingHot;
    if (trendingIntensity >= 40) return styles.trendingWarm;
    return styles.trendingMild;
  };

  return (
    <div className={styles.card}>
      {/* Trending Badge */}
      <div className={`${styles.trendingBadge} ${getTrendingBadgeClass()}`}>
        <span className={styles.fireIcon}>🔥</span>
        <span className={styles.trendingText}>Trending</span>
        <span className={styles.trendingScore}>{trendingIntensity}%</span>
      </div>

      {/* Card Header */}
      <div className={styles.cardHeader}>
        <h3 className={styles.title}>{title || 'Startup Name'}</h3>
        <div className={styles.metaTags}>
          <span className={styles.categoryTag}>{category || 'Uncategorized'}</span>
          {type && <span className={styles.typeTag}>{type}</span>}
        </div>
      </div>

      {/* Description */}
      <p className={styles.description}>
        {description ? (description.length > 150 ? `${description.substring(0, 150)}...` : description) : 'No description available'}
      </p>

      {/* Metrics Section */}
      <div className={styles.metricsSection}>
        <div className={styles.metricRow}>
          <div className={styles.metric}>
            <span className={styles.metricLabel}>Trending Score</span>
            <div className={styles.scoreBar}>
              <div 
                className={styles.scoreFill} 
                style={{ width: `${trendingIntensity}%` }}
              />
            </div>
            <span className={styles.metricValue}>{trendingIntensity}%</span>
          </div>
        </div>

        <div className={styles.metricRow}>
          <div className={styles.metric}>
            <span className={styles.metricLabel}>Popularity</span>
            <div className={styles.scoreBar}>
              <div 
                className={`${styles.scoreFill} ${styles.popularityFill}`} 
                style={{ width: `${popularityIntensity}%` }}
              />
            </div>
            <span className={styles.metricValue}>{popularityIntensity}%</span>
          </div>
        </div>

        {/* Activity Stats */}
        <div className={styles.activityStats}>
          <div className={styles.activityItem}>
            <span className={styles.activityIcon}>👁️</span>
            <span className={styles.activityValue}>{view_count_24h}</span>
            <span className={styles.activityLabel}>views (24h)</span>
          </div>
          <div className={styles.activityItem}>
            <span className={styles.activityIcon}>📊</span>
            <span className={styles.activityValue}>{view_count_7d}</span>
            <span className={styles.activityLabel}>views (7d)</span>
          </div>
          <div className={styles.activityItem}>
            <span className={styles.activityIcon}>📝</span>
            <span className={styles.activityValue}>{application_count_7d}</span>
            <span className={styles.activityLabel}>applications</span>
          </div>
          <div className={styles.activityItem}>
            <span className={styles.activityIcon}>❤️</span>
            <span className={styles.activityValue}>{favorite_count_7d}</span>
            <span className={styles.activityLabel}>favorites</span>
          </div>
        </div>

        {/* Velocity Indicator */}
        {velocity_score > 0 && (
          <div className={styles.velocityIndicator}>
            <span className={styles.velocityLabel}>Growth Velocity:</span>
            <span className={`${styles.velocityValue} ${velocity_score > 1 ? styles.velocityUp : styles.velocityDown}`}>
              {velocity_score > 1 ? '📈' : '📉'} {velocity_score.toFixed(2)}x
            </span>
          </div>
        )}

        {/* Active Positions */}
        {active_positions_count > 0 && (
          <div className={styles.positionsBadge}>
            <span className={styles.positionsIcon}>💼</span>
            <span>{active_positions_count} Active Position{active_positions_count !== 1 ? 's' : ''}</span>
          </div>
        )}
      </div>

      {/* View Details Button */}
      <Link 
        to={getStartupDetailPath(id)} 
        className={styles.viewDetailsButton}
        onClick={() => {
          console.log('🔗 [TrendingStartupCard] Navigating to:', getStartupDetailPath(id));
          console.log('🔗 [TrendingStartupCard] Startup ID:', id);
        }}
      >
        View Details
      </Link>
    </div>
  );
};

TrendingStartupCard.propTypes = {
  startup: PropTypes.shape({
    id: PropTypes.string.isRequired,
    title: PropTypes.string,
    description: PropTypes.string,
    category: PropTypes.string,
    type: PropTypes.string,
    trending_score: PropTypes.number,
    popularity_score: PropTypes.number,
    velocity_score: PropTypes.number,
    view_count_24h: PropTypes.number,
    view_count_7d: PropTypes.number,
    application_count_7d: PropTypes.number,
    favorite_count_7d: PropTypes.number,
    active_positions_count: PropTypes.number,
  }).isRequired,
};

export default TrendingStartupCard;

