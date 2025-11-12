import React from "react";
import { Navbar } from "../../components/Navbar/Navbar";
import { Footer } from "../../components/Footer/Footer";
import { Link } from "react-router-dom";
import styles from "./LandingPage.module.css"; // Assuming you're using CSS Modules
import { LeftImageRightText } from "../../components/LeftImageRightText/LeftImageRightText";
import { RightTextLeftImage } from "../../components/RightTextLeftImage/RightTextLeftImage";
import { AnimatedStats } from "../../components/AnimatedStats copy/AnimatedStats";
import { Submit } from "../../components/Submit/Submit";

function LandingPage() {
  return (
    <>
      <Navbar />

      <div className={styles.home_container}>
        <div className={styles.hero_section}>
          {/* Overlay */}
          <div className={styles.overlay}></div>

          {/* Content */}
          <div className={styles.hero_section_content}>
            <p className={styles.p1}>
            Empowering Startups Through Connection
            </p>
            <p className={styles.p2}>
            Join a vibrant network of entrepreneurs, students, and investors. Build teams, pitch ideas, and collaborate to turn innovation into real ventures.
            </p>

            <div className={styles.button_container}>
              <Link to="/collaboration">
                <button>Join a Startup</button>
              </Link>
              <Link to="/createstartup">
                <button>Upload Startup</button>
              </Link>
            </div>
          </div>
        </div>
      </div>

      <div className={styles.secondsec}>
        <img src="./stats.png" alt="" />
      </div>

      <LeftImageRightText />
      <AnimatedStats />
      <RightTextLeftImage />

    <div className={styles.bottombanner}>
    <h1>Join 500k+ founders and buyers already doing business on StartLink</h1>
    <Link to="/signup">
    <Submit btn1text={"Join Now"}/>  
    </Link>
    </div>

      <Footer />
    </>
  );
}

export { LandingPage };
