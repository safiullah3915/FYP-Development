// Updated component to support both startup object and individual props
import React from "react";
import styles from "./MarketPlaceCard.module.css";
import { Link } from "react-router-dom";
import { getStartupDetailPath } from "../../utils/idUtils";

const MarketPlaceCard = ({ 
  startup, 
  id, 
  title, 
  description, 
  revenue, 
  profit, 
  asking_price, 
  category, 
  type, 
  field,
  recommendationContext, // Optional: { sessionId, rank, score, method }
  ...rest 
}) => {
  // If startup object is passed, use it directly; otherwise construct from individual props
  // Also merge with rest props in case there are additional fields from the API
  const startupData = startup || {
    id,
    title,
    description,
    revenue,
    profit,
    asking_price,
    category,
    type,
    field,
    ...rest
  };
  
  // Debug logging
  console.log('[MarketPlaceCard] All props:', { startup, id, title, description, revenue, profit, asking_price, category, type, field, rest });
  console.log('[MarketPlaceCard] Final startupData:', startupData);
  console.log('[MarketPlaceCard] Revenue value:', startupData.revenue);
  console.log('[MarketPlaceCard] Profit value:', startupData.profit);
  console.log('[MarketPlaceCard] Asking Price value:', startupData.asking_price);
  
  // Build link destination with recommendation context if available
  const getLinkDestination = () => {
    const basePath = getStartupDetailPath(startupData.id);
    if (recommendationContext?.sessionId) {
      const params = new URLSearchParams({
        recommendation_session_id: recommendationContext.sessionId,
      });
      if (recommendationContext.rank) {
        params.append('recommendation_rank', recommendationContext.rank.toString());
      }
      return `${basePath}?${params.toString()}`;
    }
    return basePath;
  };

  return (
    <Link to={getLinkDestination()} className={styles.linkWrapper}>
    <div className={styles.card}>

      <div className={styles.cardHeader}>
        <div className={styles.icon}>
            <img src="/diamond.svg" alt="" />
            <img src="/Decentralized Network.svg" alt="" />
        <h3>{startupData.title || 'Startup Name'}</h3>
        </div>
        <span className={styles.tag}>{startupData.category || 'Marketplace'}</span>
      </div>
      <p className={styles.description}>{startupData.description || 'No description available'}</p>
      <div className={styles.stats}>
        <div className={styles.statsheading}>
          Revenue
          <p>{startupData?.revenue && startupData.revenue !== '' ? startupData.revenue : '$0'}</p>
        </div>
        <div className={styles.statsheading}>
          Profit
          <p>{startupData?.profit && startupData.profit !== '' ? startupData.profit : '$0'}</p>
        </div>
        <div className={styles.statsheading}>
          Asking Price
          <p>{startupData?.asking_price && startupData.asking_price !== '' ? startupData.asking_price : '$0'}</p>
        </div>
      </div>
    </div>
    </Link>
  );
};

export default MarketPlaceCard;
