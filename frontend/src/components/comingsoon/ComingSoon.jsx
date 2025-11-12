import { useState, useEffect } from "react";
import styles from "./ComingSoon.module.css";
import { Navbar } from "../Navbar/Navbar";

function ComingSoon() {
  const [timeLeft, setTimeLeft] = useState({ days: 0, hours: 0, minutes: 0, seconds: 0 });

  useEffect(() => {
    const targetDate = new Date("2025-12-31T23:59:59"); // ⏳ set your launch date

    const interval = setInterval(() => {
      const now = new Date();
      const difference = targetDate - now;

      if (difference <= 0) {
        clearInterval(interval);
        setTimeLeft({ days: 0, hours: 0, minutes: 0, seconds: 0 });
      } else {
        setTimeLeft({
          days: Math.floor(difference / (1000 * 60 * 60 * 24)),
          hours: Math.floor((difference / (1000 * 60 * 60)) % 24),
          minutes: Math.floor((difference / 1000 / 60) % 60),
          seconds: Math.floor((difference / 1000) % 60),
        });
      }
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  return (
    <>
  
    <Navbar/>
    <div className={styles.comingSoon}>
      <h1 className={styles.title}>✨ Coming Soon ✨</h1>
      <p className={styles.subtitle}>We are working hard to bring something amazing for you.</p>

      <div className={styles.countdown}>
        <div><span>{timeLeft.days}</span><small>Days</small></div>
        <div><span>{timeLeft.hours}</span><small>Hours</small></div>
        <div><span>{timeLeft.minutes}</span><small>Minutes</small></div>
        <div><span>{timeLeft.seconds}</span><small>Seconds</small></div>
      </div>

      <form className={styles.notifyForm}>
        <input type="email" placeholder="Enter your email" />
        <button type="submit">Notify Me</button>
      </form>
    </div>
      </>
  );
}

export default ComingSoon;
