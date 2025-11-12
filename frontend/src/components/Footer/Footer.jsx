import React from "react";
import styles from "./Footer.module.css";

const Footer = () => {
  return (
    <footer className={styles.footer}>
      <div className={styles.footer_content}>
        <div className={styles.footer_section}>
          <div className={styles.social_icons}>
            <a
              href="https://facebook.com"
              target="_blank"
              rel="noopener noreferrer"
              className={styles.icon}
            >
              <img
                className={styles.icons}
                src="/facebook_icon.png"
                alt="Facebook"
              />
            </a>
            <a
              href="https://instagram.com"
              target="_blank"
              rel="noopener noreferrer"
              className={styles.icon}
            >
              <img
                className={styles.icons}
                src="/Instagram.png"
                alt="Instagram"
              />
            </a>
          </div>
        </div>

        <div className={styles.links_container}>
          <div className={styles.link_group}>
            <h3>Company</h3>
            <ul>
              <li>
                <a href="/about">About Us</a>
              </li>
              <li>
                <a href="/services">Services</a>
              </li>
              <li>
                <a href="/contact">Contact</a>
              </li>
            </ul>
          </div>

          <div className={styles.link_group}>
            <h3>Services</h3>
            <ul>
              <li>
                <a href="/services">Solar Solutions</a>
              </li>
              <li>
                <a href="/services">Energy Management</a>
              </li>
              <li>
                <a href="/services">Consulting</a>
              </li>
            </ul>
          </div>

          <div className={styles.link_group}>
            <h3>Resources</h3>
            <ul>
              <li>
                <a href="/about">Our Projects</a>
              </li>
              <li>
                <a href="/contact">Support</a>
              </li>
              <li>
                <a href="/contact">FAQ</a>
              </li>
            </ul>
          </div>
        </div>
      </div>

      <div className={styles.footer_bottom}>
        <p>&copy; 2025 StartLink. All rights reserved. Product by SpectraMind AI</p>
      </div>
    </footer>
  );
};

export { Footer };
