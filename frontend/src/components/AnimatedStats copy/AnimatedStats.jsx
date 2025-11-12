import React, { useState, useEffect, useRef } from "react";
import styles from "./AnimatedStats.module.css";

const AnimatedStats = () => {
  const [stats, setStats] = useState({
    startups: 0,
    sold: 0,
    volume: 0,
    deals: 0,
  });
  const [isVisible, setIsVisible] = useState(false);
  const statsRef = useRef(null);

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsVisible(true);
        }
      },
      { threshold: 0.3 }
    );

    if (statsRef.current) {
      observer.observe(statsRef.current);
    }

    return () => {
      if (statsRef.current) {
        observer.unobserve(statsRef.current);
      }
    };
  }, []);

  useEffect(() => {
    if (isVisible) {
      const intervals = {
        startups: setInterval(() => {
          setStats((prev) => {
            if (prev.startups < 100) return { ...prev, startups: prev.startups + 2 };
            clearInterval(intervals.startups);
            return { ...prev, startups: 100 };
          });
        }, 30),

        sold: setInterval(() => {
          setStats((prev) => {
            if (prev.sold < 500) return { ...prev, sold: prev.sold + 10 };
            clearInterval(intervals.sold);
            return { ...prev, sold: 500 };
          });
        }, 30),

        volume: setInterval(() => {
          setStats((prev) => {
            if (prev.volume < 100) return { ...prev, volume: prev.volume + 2 };
            clearInterval(intervals.volume);
            return { ...prev, volume: 100 };
          });
        }, 30),

        deals: setInterval(() => {
          setStats((prev) => {
            if (prev.deals < 100) return { ...prev, deals: prev.deals + 2 };
            clearInterval(intervals.deals);
            return { ...prev, deals: 100 };
          });
        }, 30),
      };

      return () => {
        Object.values(intervals).forEach(clearInterval);
      };
    }
  }, [isVisible]);

  return (
    <div
      ref={statsRef}
      className={`${styles.stats_section} ${isVisible ? styles.visible : ""}`}
    >
      <div className={styles.stats_container}>
        <div className={styles.stat_item}>
          <p className={styles.stat_value}>{stats.startups}+</p>
          <p className={styles.stat_label}>Startups</p>
        </div>
        <div className={styles.stat_item}>
          <p className={styles.stat_value}>{stats.sold}+</p>
          <p className={styles.stat_label}>Startups Sold</p>
        </div>
        <div className={styles.stat_item}>
          <p className={styles.stat_value}>{stats.volume}+</p>
          <p className={styles.stat_label}>Volume</p>
        </div>
        <div className={styles.stat_item}>
          <p className={styles.stat_value}>{stats.deals}+</p>
          <p className={styles.stat_label}>Closed Deals</p>
        </div>
      </div>
    </div>
  );
};

export {AnimatedStats};
