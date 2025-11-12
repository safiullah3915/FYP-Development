import React from "react";
import { Link } from "react-router-dom";
import styles from "./TrendingStartupCard.module.css";

const TrendingStartupCard = ({ startup }) => {
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
      <Link to={`/startupdetail/${id}`} className={styles.viewDetailsButton}>
        View Details
      </Link>
    </div>
  );
};

export default TrendingStartupCard;

