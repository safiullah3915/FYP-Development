import React from "react";
import styles from "./CollaborationCard.module.css";
import { Link } from 'react-router-dom';

const CollaborationCard = ({ id, title, description, earn_through, phase, team_size, category, type }) => {
  // Always navigate to startup detail page
  const getLinkDestination = () => {
      return `/startupdetail/${id}`;
  };

  return (
    <Link to={getLinkDestination()} className={styles.linkWrapper}>
      <div className={styles.card}>
        <div className={styles.cardHeader}>
          <div className={styles.icon}>
            <img src="/diamond.svg" alt="" />
            <img src="/Decentralized Network.svg" alt="" />
            <h3>{title || 'Startup Name'}</h3>
          </div>
          <span className={styles.tag}>{category || 'Collaboration'}</span>
        </div>
        <p className={styles.description}>{description || 'No description available'}</p>
        <div className={styles.stats}>
          <div className={styles.statsheading}>
            Earn Through
            <p>{earn_through || 'Equity'}</p>
          </div>
          <div className={styles.statsheading}>
            Category
            <p>{category || 'SaaS'}</p>
          </div>
          <div className={styles.statsheading}>
            Team Size
            <p>{team_size || '1-5'}</p>
          </div>
        </div>
      </div>
    </Link>
  );
};

export default CollaborationCard;
