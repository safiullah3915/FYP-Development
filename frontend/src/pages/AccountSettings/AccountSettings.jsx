import React, { useState, useEffect } from "react";
import { useAuth } from '../../contexts/AuthContext';
import { userAPI } from '../../utils/apiServices';
import { toast } from 'react-toastify';
import styles from "./AccountSettings.module.css";



export default function AccountSettings() {
  const { user, loading: userLoading } = useAuth();
  const [loading, setLoading] = useState(true);
  const [profilePublic, setProfilePublic] = useState(false);
  const [profileData, setProfileData] = useState({});

  useEffect(() => {
    loadProfileData();
  }, []);

  const loadProfileData = async () => {
    try {
      setLoading(true);
      const response = await userAPI.getProfileData();
      const data = response.data;
      
      if (data.profile) {
        setProfilePublic(data.profile.is_public || false);
        setProfileData(data.profile);
      }
    } catch (error) {
      console.error('Failed to load profile:', error);
      toast.error('Failed to load profile');
    } finally {
      setLoading(false);
    }
  };

  const saveProfile = async () => {
    try {
      const profileData = {
        is_public: profilePublic,
      };
      
      await userAPI.updateProfileData(profileData);
      toast.success('Profile updated successfully');
    } catch (error) {
      console.error('Failed to save profile:', error);
      toast.error('Failed to save profile');
    }
  };

  return (
    <div className={styles.container}>
      {/* Main Content Only (no sidebar) */}
      <main className={styles.main}>
        <div className={styles.section}>
          <h4>Make profile public</h4>
          <div className={styles.toggleRow}>
            <span>Off</span>
            <label className={styles.switch}>
              <input
                type="checkbox"
                checked={profilePublic}
                onChange={() => setProfilePublic(!profilePublic)}
              />
              <span className={styles.slider}></span>
            </label>
            <span>On</span>
          </div>
        </div>

        {/* Profile Card */}
        <div className={styles.profileCard}>
          <div className={styles.avatar}>
            {user?.username?.charAt(0)?.toUpperCase() || 'U'}
          </div>
          <h2>{user?.username || 'Unknown User'}</h2>
          <p className={styles.location}>{user?.email || 'No email'} • {user?.role || 'No role'}</p>
          <p className={styles.tldr}>
            {loading ? 'Loading profile...' : 
             (user ? `${user.username} is a ${user.role} on our platform.` : 
              'Profile information not available.')}
          </p>

          <div className={styles.skillSection}>
            <h4>Top Skill</h4>
            <p className={styles.skill}>Programming (Python)</p>
            <p className={styles.skillDesc}>
              I've been building backend solutions with Python for over 2 years.
            </p>
            <a href="#">+ Add Reference</a>
          </div>

          <div className={styles.extraSection}>
            <h4>Skills & Hobbies</h4>
            <button className={styles.addBtn}>+ Add skill</button>
          </div>

          <div className={styles.extraSection}>
            <h4>About me</h4>
            <button className={styles.addBtn}>+ Add Business Experience</button>
          </div>

          <div className={styles.footerRow}>
            <button 
              className={styles.saveBtn}
              onClick={saveProfile}
              disabled={loading}
            >
              {loading ? 'Saving...' : 'Save & Exit'}
            </button>
          </div>
        </div>
      </main>
    </div>
  );
}
