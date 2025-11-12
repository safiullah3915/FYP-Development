import React, { useState, useEffect } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import styles from "./Navbar.module.css";
import { useAuth } from "../../contexts/AuthContext";
import MessageService from "../../services/MessageService";
import { notificationAPI } from "../../utils/apiServices";

const Navbar = () => {
  const [isScrolled, setIsScrolled] = useState(false);
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const location = useLocation();
  const navigate = useNavigate();
  const { user, isAuthenticated, logout, isEntrepreneur, isStudent, isInvestor } = useAuth();
  const [unreadCount, setUnreadCount] = useState(0);

  // Change navbar style on scroll
  // useEffect(() => {
  //   const handleScroll = () => {
  //     setIsScrolled(window.scrollY > 100);
  //   };

  //   window.addEventListener("scroll", handleScroll);
  //   return () => window.removeEventListener("scroll", handleScroll);
  // }, []);

  // Close mobile menu on route change
  useEffect(() => {
    setIsMenuOpen(false);
  }, [location]);

  // Prevent body scroll when menu is open
  useEffect(() => {
    document.body.style.overflow = isMenuOpen ? "hidden" : "unset";
    return () => {
      document.body.style.overflow = "unset";
    };
  }, [isMenuOpen]);

  const toggleMenu = () => {
    setIsMenuOpen(!isMenuOpen);
  };

  const closeMenu = () => {
    setIsMenuOpen(false);
  };

  const handleLogout = async () => {
    await logout();
    navigate("/");
  };

  // Unread messages indicator
  useEffect(() => {
    if (!isAuthenticated || !user) {
      setUnreadCount(0);
      return;
    }

    const storageKey = "messages:lastSeenAt";
    const getLastSeen = () => {
      const v = localStorage.getItem(storageKey);
      const n = Number(v);
      return Number.isFinite(n) ? n : 0;
    };

    const markSeenIfOnMessages = () => {
      if (location.pathname.startsWith("/message")) {
        localStorage.setItem(storageKey, String(Date.now()));
        setUnreadCount(0);
      }
    };

    const computeUnread = async () => {
      try {
        const [conversations, notificationsResp] = await Promise.all([
          MessageService.getConversations(),
          notificationAPI.getNotifications(),
        ]);
        const lastSeen = getLastSeen();

        // Unread messages since last seen (from other users only)
        const msgCount = (Array.isArray(conversations) ? conversations : []).reduce((acc, conv) => {
          const lm = conv?.last_message;
          if (!lm) return acc;
          const ts = new Date(lm.created_at).getTime();
          const isFromOther = lm?.sender?.id && user?.id && lm.sender.id !== user.id;
          return acc + (isFromOther && ts > lastSeen ? 1 : 0);
        }, 0);

        // Unread interest-related notifications since last seen
        const notifications = Array.isArray(notificationsResp?.data) ? notificationsResp.data : notificationsResp?.data?.results || [];
        const interestTypes = new Set(["new_application", "interest", "new_interest"]);
        const interestCount = notifications.reduce((acc, n) => {
          const ts = n?.created_at ? new Date(n.created_at).getTime() : 0;
          const isInterest = interestTypes.has(n?.type);
          const isUnread = n?.is_read === false;
          return acc + (isInterest && isUnread && ts > lastSeen ? 1 : 0);
        }, 0);

        setUnreadCount(msgCount + interestCount);
      } catch (e) {
        // silent fail
      }
    };

    // initial mark if on messages page
    markSeenIfOnMessages();
    // initial fetch
    computeUnread();
    // poll every 5s for snappier updates
    const interval = setInterval(computeUnread, 5000);

    // also recompute on route change
    computeUnread();

    return () => clearInterval(interval);
  }, [isAuthenticated, user, location.pathname]);

  // Role-based navigation links
  const getNavLinks = () => {
    const baseLinks = [];

    if (isAuthenticated) {
      baseLinks.push(
        { path: "/dashboard", label: "Dashboard" },
        { path: "/message", label: "Messages" },
        { path: "/account", label: "Account" }
      );

      if (isInvestor()) {
        baseLinks.splice(1, 0, { path: "/marketplace", label: "Marketplace" });
      }

      // Role-specific links
      if (isEntrepreneur()) {
        baseLinks.push(
          { path: "/collaboration", label: "Collaboration" },
          { path: "/createstartup", label: "Create Startup" }
        );
      }
      
      if (isStudent()) {
        baseLinks.push(
          { path: "/collaboration", label: "Collaboration" }
        );
      }
    } else {
      baseLinks.push(
        { path: "/", label: "Home" },
        { path: "/login", label: "Login" },
        { path: "/signup", label: "Signup" }
      );
    }

    return baseLinks;
  };

  const navLinks = getNavLinks();

  return (
    <nav className={`${styles.navbar} ${isScrolled ? styles.scrolled : ""}`}>
      {isAuthenticated ? (
        <div className={styles.logoContainer}>
          <img src="/logo.png" alt="Logo" className={styles.logo} />
        </div>
      ) : (
      <Link to="/" className={styles.logoContainer}>
          <img src="/logo.png" alt="Logo" className={styles.logo} />
      </Link>
      )}

      <div className={styles.navLinks}>
        {navLinks.map((link) => {
          const isMessages = link.path === "/message";
          const isActive = location.pathname === link.path;
          return (
            <Link
              key={link.path}
              to={link.path}
              className={`${styles.navLink} ${isActive ? styles.active : ""}`}
            >
              <span className={styles.navLinkLabel}>{link.label}</span>
              {isMessages && unreadCount > 0 && (
                <span className={styles.badgeDot} aria-label="unread messages" />
              )}
            </Link>
          );
        })}
      </div>

      {isAuthenticated && (
        <div className={styles.userMenu}>
          <button onClick={handleLogout} className={styles.logoutBtn}>
            Logout
          </button>
        </div>
      )}

      <button
        className={`${styles.mobileMenuButton} ${
          isMenuOpen ? styles.active : ""
        }`}
        onClick={toggleMenu}
        aria-label="Toggle menu"
      >
        <span></span>
        <span></span>
        <span></span>
      </button>

      <div
        className={`${styles.mobileMenu} ${isMenuOpen ? styles.active : ""}`}
      >
        <div className={styles.mobileMenuHeader}>
          {isAuthenticated ? (
            <div className={styles.logoContainer}>
              <img src="/logo.svg" alt="Logo" className={styles.logo} />
            </div>
          ) : (
          <Link to="/" onClick={closeMenu}>
              <img src="/logo.svg" alt="Logo" className={styles.logo} />
          </Link>
          )}
          <button className={styles.closeButton} onClick={closeMenu}>
            <span></span>
            <span></span>
          </button>
        </div>
        {navLinks.map((link) => (
          <Link
            key={link.path}
            to={link.path}
            className={`${styles.navLink} ${
              location.pathname === link.path ? styles.active : ""
            }`}
            onClick={closeMenu}
          >
            {link.label}
          </Link>
        ))}
        
        {isAuthenticated && (
          <div className={styles.mobileUserMenu}>
            <button onClick={() => { handleLogout(); closeMenu(); }} className={styles.mobileLogoutBtn}>
              Logout
            </button>
          </div>
        )}
      </div>
    </nav>
  );
};

export { Navbar };
