import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { useSpring, animated } from 'react-spring';
import {Navbar} from "../../components/Navbar/Navbar"
import { toast } from 'react-toastify';
import { useAuth } from '../../contexts/AuthContext';
import { userAPI, applicationAPI } from '../../utils/apiServices';

// This is the main component that renders the settings page.
const App = () => {
  const { user, isStudent, isEntrepreneur, isInvestor, logout } = useAuth();
  const [isProfilePublic, setIsProfilePublic] = useState(false);
  const [references, setReferences] = useState([]);
  const [skills, setSkills] = useState([]);
  const [experiences, setExperiences] = useState([]);
  const [referencesPage, setReferencesPage] = useState(1);
  const [skillsPage, setSkillsPage] = useState(1);
  const [experiencesPage, setExperiencesPage] = useState(1);
  const [newReference, setNewReference] = useState({ title: '', description: '' });
  const [newSkill, setNewSkill] = useState({ title: '', description: '' });
  const [newExperience, setNewExperience] = useState({ title: '', description: '' });
  const [editingItem, setEditingItem] = useState(null); // { type, index }
  const [editForm, setEditForm] = useState({ title: '', description: '' });
  const [isEditSaving, setIsEditSaving] = useState(false);
  const PAGE_SIZE = 3;

  // Application management state
  const [applications, setApplications] = useState([]);
  const [startupApplications, setStartupApplications] = useState([]);
  const [favorites, setFavorites] = useState([]);
  const [interests, setInterests] = useState([]);
  const [profileStats, setProfileStats] = useState({});
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadUserData();
  }, []);

  const loadUserData = async () => {
    try {
      setLoading(true);
      
      // Load user profile data
      const profileResponse = await userAPI.getProfileData();
      
      const data = profileResponse.data;
      
      // Update profile state
      if (data.profile) {
        setIsProfilePublic(data.profile.is_public || false);
        setSkills(data.profile.skills || []);
        setExperiences(data.profile.experience || []);
        setReferences(data.profile.references || []);
        setReferencesPage(1);
        setSkillsPage(1);
        setExperiencesPage(1);
      }
      
      // Update stats
      if (data.stats) {
        setProfileStats(data.stats);
      }
      
      // Load role-specific data
      if (isStudent()) {
        setApplications(data.applications || []);
        setFavorites(data.favorites || []);
      } else if (isEntrepreneur()) {
        setStartupApplications(data.applications || []);
      } else if (isInvestor()) {
        setFavorites(data.favorites || []);
        setInterests(data.interests || []);
      }
      
    } catch (error) {
      console.error('Failed to load user data:', error);
      toast.error('Failed to load user data');
    } finally {
      setLoading(false);
    }
  };

  const fade = useSpring({
    from: { opacity: 0 },
    to: { opacity: 1 },
    config: { duration: 500 },
  });

  // Handle the toggle for making the profile public.
  const handleToggle = () => {
    setIsProfilePublic(prev => !prev);
  };

  // Application management functions
  const handleApproveApplication = async (applicationId) => {
    try {
      await applicationAPI.approveApplication(applicationId);
      toast.success('Application approved');
      loadUserData(); // Reload data
    } catch (error) {
      console.error('Failed to approve application:', error);
      toast.error('Failed to approve application');
    }
  };

  const handleDeclineApplication = async (applicationId) => {
    try {
      await applicationAPI.declineApplication(applicationId);
      toast.success('Application declined');
      loadUserData(); // Reload data
    } catch (error) {
      console.error('Failed to decline application:', error);
      toast.error('Failed to decline application');
    }
  };

  const handleSave = async () => {
    try {
      const profileData = {
        is_public: isProfilePublic,
        skills: skills,
        experience: experiences,
        references: references
      };

      await userAPI.updateProfileData(profileData);
      
      toast.success('Profile updated successfully');
    } catch (error) {
      console.error('Failed to update profile:', error);
      toast.error('Failed to update profile');
    }
  };

  // Generic handler to add a new item to a specific list.
  const handleNewItemChange = (type, field, value) => {
    const updater = type === 'references'
      ? setNewReference
      : type === 'skills'
        ? setNewSkill
        : setNewExperience;
    updater(prev => ({ ...prev, [field]: value }));
  };

  const handleAddItem = async (type) => {
    const { title, description } =
      type === 'references'
        ? newReference
        : type === 'skills'
          ? newSkill
          : newExperience;

    if (!title.trim() || !description.trim()) {
      toast.error('Please fill in both name and description before adding.');
      return;
    }

    const newItem = { title: title.trim(), description: description.trim() };

    let updatedReferences = references;
    let updatedSkills = skills;
    let updatedExperiences = experiences;

    try {
    if (type === 'references') {
        updatedReferences = [...references, newItem];
        setReferences(updatedReferences);
        setReferencesPage(Math.ceil(updatedReferences.length / PAGE_SIZE));
        setNewReference({ title: '', description: '' });
    } else if (type === 'skills') {
        updatedSkills = [...skills, newItem];
        setSkills(updatedSkills);
        setSkillsPage(Math.ceil(updatedSkills.length / PAGE_SIZE));
        setNewSkill({ title: '', description: '' });
      } else {
        updatedExperiences = [...experiences, newItem];
        setExperiences(updatedExperiences);
        setExperiencesPage(Math.ceil(updatedExperiences.length / PAGE_SIZE));
        setNewExperience({ title: '', description: '' });
      }

      await persistProfileCollections(updatedReferences, updatedSkills, updatedExperiences);
      toast.success('Entry added successfully');
    } catch (error) {
      console.error('Failed to add entry:', error);
      toast.error('Failed to add entry');
      loadUserData();
    }
  };

  const persistProfileCollections = useCallback(async (updatedReferences, updatedSkills, updatedExperiences) => {
    const payload = {
      is_public: isProfilePublic,
      references: updatedReferences,
      skills: updatedSkills,
      experience: updatedExperiences
    };

    await userAPI.updateProfileData(payload);
  }, [isProfilePublic]);

  const handleRemoveItem = async (type, index) => {
    const confirmed = window.confirm('Are you sure you want to delete this entry? This action cannot be undone.');
    if (!confirmed) {
      return;
    }

    let updatedReferences = references;
    let updatedSkills = skills;
    let updatedExperiences = experiences;

    try {
    if (type === 'references') {
        updatedReferences = references.filter((_, i) => i !== index);
        setReferences(updatedReferences);
        setReferencesPage(prev => Math.min(prev, Math.max(1, Math.ceil(updatedReferences.length / 3))));
    } else if (type === 'skills') {
        updatedSkills = skills.filter((_, i) => i !== index);
        setSkills(updatedSkills);
        setSkillsPage(prev => Math.min(prev, Math.max(1, Math.ceil(updatedSkills.length / 3))));
      } else {
        updatedExperiences = experiences.filter((_, i) => i !== index);
        setExperiences(updatedExperiences);
        setExperiencesPage(prev => Math.min(prev, Math.max(1, Math.ceil(updatedExperiences.length / 3))));
      }

      await persistProfileCollections(updatedReferences, updatedSkills, updatedExperiences);
      toast.success('Entry removed successfully');
    } catch (error) {
      console.error('Failed to remove entry:', error);
      toast.error('Failed to remove entry');
      // Reload the latest data to keep UI in sync with the server
      loadUserData();
    }
  };

  const handleEditItem = (type, index) => {
    const getter = type === 'references' ? references : type === 'skills' ? skills : experiences;
    const currentItem = getter[index];
    if (!currentItem) {
      return;
    }

    setEditingItem({ type, index });
    setEditForm({
      title: currentItem.title || '',
      description: currentItem.description || ''
    });
  };

  const handleEditFieldChange = (field, value) => {
    setEditForm(prev => ({ ...prev, [field]: value }));
  };

  const closeEditModal = () => {
    if (isEditSaving) return;
    setEditingItem(null);
    setEditForm({ title: '', description: '' });
  };

  const handleSaveEdit = async (event) => {
    event.preventDefault();
    if (!editingItem) return;

    const title = editForm.title.trim();
    const description = editForm.description.trim();

    if (!title || !description) {
      toast.error('Both fields are required.');
      return;
    }

    setIsEditSaving(true);

    let updatedReferences = references;
    let updatedSkills = skills;
    let updatedExperiences = experiences;

    try {
      if (editingItem.type === 'references') {
        updatedReferences = references.map((item, i) =>
          i === editingItem.index ? { ...item, title, description } : item
        );
        setReferences(updatedReferences);
      } else if (editingItem.type === 'skills') {
        updatedSkills = skills.map((item, i) =>
          i === editingItem.index ? { ...item, title, description } : item
        );
        setSkills(updatedSkills);
      } else {
        updatedExperiences = experiences.map((item, i) =>
          i === editingItem.index ? { ...item, title, description } : item
        );
        setExperiences(updatedExperiences);
      }

      await persistProfileCollections(updatedReferences, updatedSkills, updatedExperiences);
      toast.success('Entry updated successfully');
      closeEditModal();
    } catch (error) {
      console.error('Failed to update entry:', error);
      toast.error('Failed to update entry');
      loadUserData();
    } finally {
      setIsEditSaving(false);
    }
  };

  const paginateList = useCallback((list, page) => {
    const start = (page - 1) * PAGE_SIZE;
    return list.slice(start, start + PAGE_SIZE);
  }, []);

  const referencesPaginated = useMemo(() => paginateList(references, referencesPage), [references, referencesPage, paginateList]);
  const skillsPaginated = useMemo(() => paginateList(skills, skillsPage), [skills, skillsPage, paginateList]);
  const experiencesPaginated = useMemo(() => paginateList(experiences, experiencesPage), [experiences, experiencesPage, paginateList]);

  const renderPagination = (page, setPage, totalItems) => {
    const totalPages = Math.max(1, Math.ceil(totalItems / PAGE_SIZE));
    if (totalPages <= 1) return null;

    const goPrev = () => setPage(prev => Math.max(1, prev - 1));
    const goNext = () => setPage(prev => Math.min(totalPages, prev + 1));

    return (
      <div className="table-pagination">
        <button type="button" className="pagination-button" onClick={goPrev} disabled={page === 1}>
          ‹
        </button>
        <span className="pagination-status">{page} / {totalPages}</span>
        <button type="button" className="pagination-button" onClick={goNext} disabled={page === totalPages}>
          ›
        </button>
      </div>
    );
  };

  return (
    <>
    
      <style>
        {`
        body {
          background-color: #111827;
          color: #F3F4F6;
          font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, "Noto Sans", sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol", "Noto Color Emoji";
          margin: 0;
          padding: 0;
        }
        .container {
          min-height: 100vh;
          display: flex;
          justify-content: center;
          align-items: center;
          padding: 1rem;
        }
        @media (min-width: 640px) {
          .container {
            padding: 2rem;
          }
        }
        .main-card {
          background-color: #1F2937;
          border-radius: 1.5rem;
          box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
          padding: 1.5rem;
          width: 100%;
          max-width: 56rem;
          display: flex;
          flex-direction: column;
          gap: 2rem;
        }
        @media (min-width: 640px) {
          .main-card {
            padding: 2.5rem;
          }
        }
        @media (min-width: 1024px) {
          .main-card {
            flex-direction: row;
          }
        }
        .content-area {
          flex: 1 1 0%;
        }
        .inner-card {
          background-color: #111827;
          border-radius: 1rem;
          padding: 1.5rem;
          position: relative;
          display: flex;
          flex-direction: column;
          align-items: center;
          border: 1px solid #374151;
        }
        @media (min-width: 1024px) {
          .inner-card {
            padding: 2.5rem;
          }
        }
        .public-profile-section {
          width: 100%;
          border-bottom: 1px solid #374151;
          padding-bottom: 2rem;
          margin-bottom: 2rem;
        }
        .public-profile-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
        }
        .public-profile-title {
          font-weight: 600;
        }
        .toggle-switch {
          position: relative;
          display: inline-block;
          width: 3.5rem;
          height: 1.5rem;
        }
        .toggle-switch input {
          opacity: 0;
          width: 0;
          height: 0;
        }
        .slider {
    position: absolute;
    cursor: pointer;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background-color: #374151;
    transition: .4s;
    height: 1.5rem;
    margin-top:2rem;
    width: 3.5rem;
    border-radius: 9999px;
        }
        .slider:before {
          position: absolute;
          content: "";
          height: 1.25rem;
          width: 1.25rem;
          left: 0.125rem;
          bottom: 0.125rem;
          background-color: white;
          transition: .4s;
          border-radius: 9999px;
        }
        input:checked + .slider {
          background-color: #8B5CF6;
        }
        input:checked + .slider:before {
          transform: translateX(2rem);
        }
        .toggle-text {
          position: absolute;
          top: 50%;
          transform: translateY(-50%);
          right: 4rem;
          font-weight: 600;
          color: #9CA3AF;
        }
        .toggle-text.on {
          right: -2rem;
          color: white;
        }
        .toggle-options {
          display: flex;
          flex-wrap: wrap;
          gap: 1rem;
          margin-top: 1.5rem;
        }
        .region-checkbox-label {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          cursor: pointer;
          color: #9CA3AF;
        }
        .region-checkbox {
          width: 1rem;
          height: 1rem;
          border-radius: 0.25rem;
          border: 1px solid #4B5563;
          background-color: #111827;
          transition: background-color 0.15s;
        }
        .region-checkbox:checked {
          background-color: #8B5CF6;
          border-color: #8B5CF6;
        }
        .region-checkbox:focus {
          outline: none;
          box-shadow: 0 0 0 2px #A855F7;
        }

        .user-profile {
          width: 100%;
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 1.5rem;
        }
        @media (min-width: 640px) {
          .user-profile {
            flex-direction: row;
          }
        }
        .profile-picture {
          position: relative;
          width: 6rem;
          height: 6rem;
          border-radius: 9999px;
          overflow: hidden;
          border: 2px solid #8B5CF6;
        }
        @media (min-width: 640px) {
          .profile-picture {
            width: 8rem;
            height: 8rem;
          }
        }
        .profile-picture img {
          width: 100%;
          height: 100%;
          object-fit: cover;
        }
        .edit-overlay {
          position: absolute;
          top: 0;
          right: 0;
          bottom: 0;
          left: 0;
          background-color: rgba(17, 24, 39, 0.5);
          display: flex;
          align-items: center;
          justify-content: center;
          opacity: 0;
          transition: opacity 0.3s;
          cursor: pointer;
        }
        .profile-picture:hover .edit-overlay {
          opacity: 1;
        }
        .profile-info {
          flex: 1 1 0%;
          text-align: center;
        }
        @media (min-width: 640px) {
          .profile-info {
            text-align: left;
          }
        }
        .profile-header {
          display: flex;
          justify-content: center;
          align-items: center;
          width: 100%;
        }
        @media (min-width: 640px) {
          .profile-header {
            justify-content: space-between;
          }
        }
        .profile-header h2 {
          font-size: 1.5rem;
          font-weight: 700;
          line-height: 2rem;
          color:#fff;
          margin: 0;
        }
        .logout-button {
          display: none;
          align-items: center;
          color: #A855F7;
          transition: color 0.15s;
        }
        .logout-button:hover {
          color: #C084FC;
        }
        @media (min-width: 640px) {
          .logout-button {
            display: inline-flex;
          }
        }
        .logout-button svg {
          height: 1rem;
          width: 1rem;
          margin-left: 0.5rem;
        }
        .profile-info p {
          color: #9CA3AF;
          font-size: 0.875rem;
          line-height: 1.25rem;
          margin: 0;
        }
        .profile-description {
          margin-top: 1rem;
          font-size: 0.875rem;
          line-height: 1.625;
          color: #D1D5DB;
        }
        .profile-description p {
          font-weight: 600;
          margin: 0;
        }
        .section-divider {
          width: 100%;
          margin-top: 2rem;
          border-top: 1px solid #374151;
          padding-top: 2rem;
        }
        .section-title {
          font-size: 1.25rem;
          line-height: 1.75rem;
          font-weight: 700;
          margin-bottom: 1rem;
        }
        .collection-form {
          display: flex;
          flex-direction: column;
          gap: 0.75rem;
          background-color: #1F2937;
          padding: 1.25rem;
          border-radius: 0.75rem;
          border: 1px solid #374151;
          margin-bottom: 1rem;
        }
        .collection-form label {
          font-size: 0.875rem;
          color: #D1D5DB;
          display: flex;
          flex-direction: column;
          gap: 0.35rem;
        }
        .collection-input {
          background-color: #F9FAFB;
          color: #1F2937;
          border: 1px solid #CBD5F5;
          border-radius: 0.5rem;
          padding: 0.6rem 0.75rem;
          font-size: 0.95rem;
          transition: border 0.2s, box-shadow 0.2s;
        }
        .collection-input:focus {
          outline: none;
          border-color: #A855F7;
          box-shadow: 0 0 0 3px rgba(168, 85, 247, 0.25);
        }
        .collection-actions {
          display: flex;
          justify-content: flex-end;
        }
        .add-button {
          display: inline-flex;
          align-items: center;
          gap: 0.4rem;
          background-color: #7C3AED;
          color: white;
          padding: 0.55rem 1.25rem;
          border-radius: 999px;
          border: none;
          font-size: 0.9rem;
          font-weight: 600;
          cursor: pointer;
          transition: background-color 0.2s, transform 0.2s;
        }
        .add-button:hover {
          background-color: #6D28D9;
          transform: translateY(-1px);
        }
        .data-table {
          width: 100%;
          border-collapse: collapse;
          background-color: #111827;
          border-radius: 0.75rem;
          overflow: hidden;
          border: 1px solid #374151;
        }
        .data-table thead {
          background-color: #1F2937;
        }
        .data-table th, .data-table td {
          padding: 0.85rem 1rem;
          text-align: left;
          color: #F9FAFB;
          border-bottom: 1px solid #374151;
          vertical-align: top;
        }
        .data-table th {
          font-weight: 600;
          color: #FFFFFF;
          font-size: 0.9rem;
        }
        .data-table tbody tr:last-child td {
          border-bottom: none;
        }
        .table-actions {
          display: flex;
          gap: 0.6rem;
        }
        .table-button {
          padding: 0.4rem 0.9rem;
          border-radius: 999px;
          font-size: 0.85rem;
          font-weight: 600;
          border: none;
          cursor: pointer;
          transition: opacity 0.2s, transform 0.2s;
        }
        .table-button:hover {
          opacity: 0.85;
          transform: translateY(-1px);
        }
        .table-button.edit {
          background-color: #2563EB;
          color: white;
        }
        .table-button.delete {
          background-color: #EF4444;
          color: white;
        }
        .table-wrapper {
          width: 100%;
          animation: tableFade 0.25s ease;
        }
        .table-pagination {
          margin-top: 0.75rem;
          width: 100%;
          display: flex;
          justify-content: center;
          align-items: center;
          gap: 1rem;
        }
        .pagination-button {
          width: 36px;
          height: 36px;
          border-radius: 50%;
          border: 1px solid #4B5563;
          background-color: transparent;
          color: #E5E7EB;
          font-size: 1.1rem;
          font-weight: 600;
          cursor: pointer;
          transition: background-color 0.2s, color 0.2s, opacity 0.2s;
          display: flex;
          align-items: center;
          justify-content: center;
          line-height: 1;
        }
        .pagination-button:hover:not(:disabled) {
          background-color: #2563EB;
          color: #FFFFFF;
        }
        .pagination-button:disabled {
          opacity: 0.4;
          cursor: not-allowed;
        }
        .pagination-status {
          color: #E5E7EB;
          font-weight: 600;
          letter-spacing: 0.05em;
        }
        @keyframes tableFade {
          from {
            opacity: 0;
            transform: translateY(6px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        .edit-modal-overlay {
          position: fixed;
          inset: 0;
          background: rgba(17, 24, 39, 0.6);
          display: flex;
          align-items: center;
          justify-content: center;
          padding: 1.5rem;
          z-index: 60;
          animation: tableFade 0.25s ease;
        }
        .edit-modal {
          background-color: #1F2937;
          border: 1px solid #374151;
          border-radius: 1rem;
          padding: 1.5rem 1rem;
          width: 100%;
          max-width: 360px;
          box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.35);
        }
        .edit-modal h4 {
          margin: 0 0 1rem 0;
          color: #F9FAFB;
          font-size: 1.15rem;
          font-weight: 600;
        }
        .edit-form {
          display: flex;
          flex-direction: column;
          gap: 0.5rem;
          width: 100%;
        }
        .edit-form label {
          display: flex;
          flex-direction: column;
          gap: 0.4rem;
          color: #E5E7EB;
          font-size: 0.9rem;
          font-weight: 600;
        }
        .edit-input {
          background-color: #111827;
          border: 1px solid #4B5563;
          border-radius: 0.6rem;
          padding: 0.6rem 0.75rem;
          color: #F9FAFB;
          font-size: 0.95rem;
          transition: border-color 0.2s, box-shadow 0.2s;
        }
        .edit-input[type="text"] {
          background-color: #111827;
        }
        .edit-input:focus {
          outline: none;
          border-color: #8B5CF6;
          box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.35);
        }
        .edit-textarea {
          min-height: 100px;
          resize: vertical;
          background-color: #111827;
        }
        .edit-modal-actions {
          display: flex;
          justify-content: flex-end;
          gap: 0.75rem;
          width: 100%;
          margin-top: 0.5rem;
          padding: 0;
        }
        .edit-button {
          padding: 0.55rem 1.25rem;
          border-radius: 999px;
          font-size: 0.9rem;
          font-weight: 600;
          border: none;
          cursor: pointer;
          transition: opacity 0.2s, transform 0.2s;
        }
        .edit-button:hover:not(:disabled) {
          opacity: 0.85;
          transform: translateY(-1px);
        }
        .edit-button:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }
        .edit-button.cancel {
          background-color: transparent;
          border: 1px solid #4B5563;
          color: #E5E7EB;
        }
        .edit-button.save {
          background-color: #7C3AED;
          color: #FFFFFF;
        }
        .save-button {
          margin-top: 2.5rem;
          width: 100%;
          padding-left: 3rem;
          padding-right: 3rem;
          padding-top: 0.75rem;
          padding-bottom: 0.75rem;
          background-color: #7C3AED;
          color: white;
          font-weight: 600;
          border-radius: 9999px;
          box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
          transition: all 0.15s;
          border: none;
          cursor: pointer;
        }
        .save-button:hover {
          background-color: #6D28D9;
          transform: scale(1.05);
        }
        @media (min-width: 1024px) {
          .save-button {
            width: auto;
          }
        }
        .input-title {
          background: #374151;
          color: #E5E7EB;
          border: none;
          border-radius: 0.5rem;
          padding: 0.5rem;
          width: 100%;
          font-size: 1.125rem;
          font-weight: 600;
          outline: none;
        }
        .remove-button {
          background: transparent;
          border: none;
          color: #EF4444;
          cursor: pointer;
          font-size: 1rem;
          font-weight: 600;
          transition: color 0.15s;
        }
        .remove-button:hover {
          color: #DC2626;
        }
        .input-description {
          background: #374151;
          color: #E5E7EB;
          border: none;
          border-radius: 0.5rem;
          padding: 0.5rem;
          width: 100%;
          font-size: 0.875rem;
          margin-top: 0.5rem;
          resize: vertical;
          outline: none;
        }
        
        /* Application Management Styles */
        .application-section {
          width: 100%;
          margin-bottom: 2rem;
          border-top: 1px solid #374151;
          padding-top: 2rem;
        }
        
        .section-title {
          color: #F3F4F6;
          font-size: 1.25rem;
          font-weight: 600;
          margin-bottom: 1rem;
        }
        
        .applications-list {
          display: flex;
          flex-direction: column;
          gap: 1rem;
        }
        
        .application-card {
          background-color: #111827;
          border: 1px solid #374151;
          border-radius: 0.75rem;
          padding: 1.5rem;
        }
        
        .application-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 0.75rem;
        }
        
        .application-header h4 {
          color: #F3F4F6;
          font-size: 1.125rem;
          font-weight: 600;
          margin: 0;
        }
        
        .status {
          padding: 0.25rem 0.75rem;
          border-radius: 0.5rem;
          font-size: 0.875rem;
          font-weight: 500;
        }
        
        .status-pending {
          background-color: #FEF3C7;
          color: #92400E;
        }
        
        .status-approved {
          background-color: #D1FAE5;
          color: #065F46;
        }
        
        .status-rejected {
          background-color: #FEE2E2;
          color: #991B1B;
        }
        
        .status-favorite {
          background-color: #FCE7F3;
          color: #BE185D;
        }
        
        .status-interest {
          background-color: #E0E7FF;
          color: #3730A3;
        }
        
        .application-description {
          color: #D1D5DB;
          font-size: 0.875rem;
          line-height: 1.5;
          margin-bottom: 0.75rem;
        }
        
        .application-date {
          color: #9CA3AF;
          font-size: 0.75rem;
          margin-bottom: 1rem;
        }
        
        .application-actions {
          display: flex;
          gap: 0.75rem;
        }
        
        .approve-btn, .decline-btn {
          padding: 0.5rem 1rem;
          border-radius: 0.5rem;
          font-size: 0.875rem;
          font-weight: 500;
          border: none;
          cursor: pointer;
          transition: all 0.2s;
        }
        
        .approve-btn {
          background-color: #10B981;
          color: white;
        }
        
        .approve-btn:hover {
          background-color: #059669;
        }
        
        .decline-btn {
          background-color: #EF4444;
          color: white;
        }
        
        .decline-btn:hover {
          background-color: #DC2626;
        }
        `}
      </style>
      <Navbar/>
      <animated.div style={fade} className="container">
        <div className="main-card">
          {/* Main content area */}
          <div className="content-area">
            <div className="inner-card">
              {/* Make Profile Public Section */}
              <div className="public-profile-section">
                <div className="public-profile-header">
                  <h3 className="public-profile-title">Make profile public</h3>
                  <div className="relative">
                    <span className={`toggle-text ${isProfilePublic ? 'on' : ''}`}>
                      {/* {isProfilePublic ? 'On' : 'Off'} */}
                    </span>
                    <label className="toggle-switch">
                      <input type="checkbox" checked={isProfilePublic} onChange={handleToggle} />
                      <span className="slider"></span>
                    </label>
                  </div>
                </div>
                <p className="text-gray-400 text-sm mt-2">
                  Make your profile discoverable to other members on the platform.
                </p>
              </div>
              {/* User Profile Section */}
              <div className="user-profile">
                <div className="profile-picture">
                  <img
                    src={`https://placehold.co/128x128/333/fff?text=${user?.username?.charAt(0)?.toUpperCase() || 'U'}`}
                    alt={`${user?.username || 'User'} profile`}
                  />
                  <div className="edit-overlay">
                    <span className="text-sm">Edit</span>
                  </div>
                </div>
                <div className="profile-info">
                  <div className="profile-header">
                    <h2>{user?.username || 'Unknown User'}</h2>
                    <button 
                      className="logout-button"
                      onClick={logout}
                      title="Logout"
                    >
                      <span>Logout</span>
                      <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1m0-16V6m6 10v-3a4 4 0 00-4 4H7a4 4 0 00-4 4v3" />
                      </svg>
                    </button>
                  </div>
                  <p>{user?.email || 'No email'} {user?.role && `• ${user.role.charAt(0).toUpperCase() + user.role.slice(1)}`}</p>
                  <div className="profile-description">
                    <p>{loading ? 'Loading profile...' : (user ? `${user.username} is a ${user.role} on our platform.` : 'No user information available.')}</p>
                    {!loading && profileStats && (
                      <div className="profile-stats" style={{ marginTop: '1rem', display: 'flex', gap: '1rem', fontSize: '0.875rem', color: '#9CA3AF' }}>
                        {isStudent() && (
                          <span>Applications: {profileStats.applications_submitted || 0}</span>
                        )}
                        {isEntrepreneur() && (
                          <span>Startups: {profileStats.startups_created || 0}</span>
                        )}
                        {isInvestor() && (
                          <span>Favorites: {profileStats.favorites_count || 0}</span>
                        )}
                        <span>Member since: {user?.created_at ? new Date(user.created_at).getFullYear() : 'Unknown'}</span>
                      </div>
                    )}
                  </div>
                </div>
              </div>
              {/* Top Skills Section */}
              <div className="section-divider">
                <h3 className="section-title">Top Skills</h3>
                <div className="collection-form">
                  <label>
                    Skill name
                      <input
                        type="text"
                      className="collection-input"
                      placeholder="e.g., Programming (Python)"
                      value={newReference.title}
                      onChange={(e) => handleNewItemChange('references', 'title', e.target.value)}
                    />
                  </label>
                  <label>
                    Description
                    <textarea
                      className="collection-input"
                      rows="3"
                      placeholder="Brief description of the skill or reference"
                      value={newReference.description}
                      onChange={(e) => handleNewItemChange('references', 'description', e.target.value)}
                    />
                  </label>
                  <div className="collection-actions">
                    <button onClick={() => handleAddItem('references')} className="add-button" type="button">
                      Add Skill
                      </button>
                    </div>
                  </div>
                {references.length > 0 && (
                  <div className="table-wrapper" key={`${referencesPage}-${references.length}`}>
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>Name</th>
                        <th>Description</th>
                        <th>Actions</th>
                      </tr>
                    </thead>
                    <tbody>
                      {referencesPaginated.map((item, index) => {
                        const globalIndex = (referencesPage - 1) * PAGE_SIZE + index;
                        return (
                        <tr key={globalIndex}>
                          <td>{item.title}</td>
                          <td>{item.description}</td>
                          <td>
                            <div className="table-actions">
                              <button
                                className="table-button edit"
                                type="button"
                                onClick={() => handleEditItem('references', globalIndex)}
                              >
                                Edit
                              </button>
                              <button
                                className="table-button delete"
                                type="button"
                                onClick={() => handleRemoveItem('references', globalIndex)}
                              >
                                Delete
                              </button>
                            </div>
                          </td>
                        </tr>
                      )})}
                    </tbody>
                  </table>
                  </div>
                )}
                {renderPagination(referencesPage, setReferencesPage, references.length)}
              </div>
              {/* Skills & Hobbies Section */}
              <div className="section-divider">
                  <h3 className="section-title">Skills & Hobbies</h3>
                <div className="collection-form">
                  <label>
                    Name
                      <input
                        type="text"
                      className="collection-input"
                      placeholder="e.g., UI Design"
                      value={newSkill.title}
                      onChange={(e) => handleNewItemChange('skills', 'title', e.target.value)}
                    />
                  </label>
                  <label>
                    Description
                    <textarea
                      className="collection-input"
                      rows="3"
                      placeholder="Share details about this skill or hobby"
                      value={newSkill.description}
                      onChange={(e) => handleNewItemChange('skills', 'description', e.target.value)}
                    />
                  </label>
                  <div className="collection-actions">
                    <button onClick={() => handleAddItem('skills')} className="add-button" type="button">
                      Add Entry
                      </button>
                    </div>
                  </div>
                {skills.length > 0 && (
                  <div className="table-wrapper" key={`${skillsPage}-${skills.length}`}>
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>Name</th>
                        <th>Description</th>
                        <th>Actions</th>
                      </tr>
                    </thead>
                    <tbody>
                      {skillsPaginated.map((item, index) => {
                        const globalIndex = (skillsPage - 1) * PAGE_SIZE + index;
                        return (
                        <tr key={globalIndex}>
                          <td>{item.title}</td>
                          <td>{item.description}</td>
                          <td>
                            <div className="table-actions">
                              <button
                                className="table-button edit"
                                type="button"
                                onClick={() => handleEditItem('skills', globalIndex)}
                              >
                                Edit
                              </button>
                              <button
                                className="table-button delete"
                                type="button"
                                onClick={() => handleRemoveItem('skills', globalIndex)}
                              >
                                Delete
                              </button>
                            </div>
                          </td>
                        </tr>
                      )})}
                    </tbody>
                  </table>
                  </div>
                )}
                {renderPagination(skillsPage, setSkillsPage, skills.length)}
              </div>
              {/* Business Experience Section */}
              <div className="section-divider">
                  <h3 className="section-title">Business Experience</h3>
                <div className="collection-form">
                  <label>
                    Company / Role
                      <input
                        type="text"
                      className="collection-input"
                      placeholder="e.g., Product Manager at StartupX"
                      value={newExperience.title}
                      onChange={(e) => handleNewItemChange('experiences', 'title', e.target.value)}
                    />
                  </label>
                  <label>
                    Description
                    <textarea
                      className="collection-input"
                      rows="3"
                      placeholder="Summarize your responsibilities or achievements"
                      value={newExperience.description}
                      onChange={(e) => handleNewItemChange('experiences', 'description', e.target.value)}
                    />
                  </label>
                  <div className="collection-actions">
                    <button onClick={() => handleAddItem('experiences')} className="add-button" type="button">
                      Add Experience
                      </button>
                    </div>
                  </div>
                {experiences.length > 0 && (
                  <div className="table-wrapper" key={`${experiencesPage}-${experiences.length}`}>
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>Name</th>
                        <th>Description</th>
                        <th>Actions</th>
                      </tr>
                    </thead>
                    <tbody>
                      {experiencesPaginated.map((item, index) => {
                        const globalIndex = (experiencesPage - 1) * PAGE_SIZE + index;
                        return (
                        <tr key={globalIndex}>
                          <td>{item.title}</td>
                          <td>{item.description}</td>
                          <td>
                            <div className="table-actions">
                              <button
                                className="table-button edit"
                                type="button"
                                onClick={() => handleEditItem('experiences', globalIndex)}
                              >
                                Edit
                              </button>
                              <button
                                className="table-button delete"
                                type="button"
                                onClick={() => handleRemoveItem('experiences', globalIndex)}
                              >
                                Delete
                              </button>
                            </div>
                          </td>
                        </tr>
                      )})}
                    </tbody>
                  </table>
                  </div>
                )}
                {renderPagination(experiencesPage, setExperiencesPage, experiences.length)}
              </div>

              {/* Application Management Sections */}
              {isStudent() && applications.length > 0 && (
                <div className="application-section">
                  <h3 className="section-title">My Applications</h3>
                  <div className="applications-list">
                    {applications.map((app) => (
                      <div key={app.id} className="application-card">
                        <div className="application-header">
                          <h4>{app.startup?.title || 'Startup'}</h4>
                          <span className={`status status-${app.status}`}>
                            {app.status}
                          </span>
                        </div>
                        <p className="application-description">
                          {app.cover_letter?.substring(0, 100)}...
                        </p>
                        <div className="application-date">
                          Applied: {new Date(app.created_at).toLocaleDateString()}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {isEntrepreneur() && startupApplications.length > 0 && (
                <div className="application-section">
                  <h3 className="section-title">Applications to My Startups</h3>
                  <div className="applications-list">
                    {startupApplications.map((app) => (
                      <div key={app.id} className="application-card">
                        <div className="application-header">
                          <h4>{app.applicant?.username || 'Applicant'}</h4>
                          <span className={`status status-${app.status}`}>
                            {app.status}
                          </span>
                        </div>
                        <p className="application-description">
                          {app.cover_letter?.substring(0, 100)}...
                        </p>
                        <div className="application-actions">
                          {app.status === 'pending' && (
                            <>
                              <button 
                                onClick={() => handleApproveApplication(app.id)}
                                className="approve-btn"
                              >
                                Approve
                              </button>
                              <button 
                                onClick={() => handleDeclineApplication(app.id)}
                                className="decline-btn"
                              >
                                Decline
                              </button>
                            </>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {isInvestor() && (
                <>
                  {favorites.length > 0 && (
                    <div className="application-section">
                      <h3 className="section-title">My Favorites</h3>
                      <div className="applications-list">
                        {favorites.map((fav) => (
                          <div key={fav.id} className="application-card">
                            <div className="application-header">
                              <h4>{fav.startup?.title || 'Startup'}</h4>
                              <span className="status status-favorite">
                                Favorited
                              </span>
                            </div>
                            <p className="application-description">
                              {fav.startup?.description?.substring(0, 100)}...
                            </p>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {interests.length > 0 && (
                    <div className="application-section">
                      <h3 className="section-title">My Interests</h3>
                      <div className="applications-list">
                        {interests.map((interest) => (
                          <div key={interest.id} className="application-card">
                            <div className="application-header">
                              <h4>{interest.startup?.title || 'Startup'}</h4>
                              <span className="status status-interest">
                                Interest Expressed
                              </span>
                            </div>
                            <p className="application-description">
                              {interest.message}
                            </p>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </>
              )}

              {/* Save Button */}
              <button
                onClick={handleSave}
                className="save-button"
              >
                Save & Exit
              </button>
            </div>
          </div>
        </div>
        {editingItem && (
          <div className="edit-modal-overlay" role="dialog" aria-modal="true">
            <div className="edit-modal">
              <h4>Edit Entry</h4>
              <form className="edit-form" onSubmit={handleSaveEdit}>
                <label>
                  Name
                  <input
                    type="text"
                    className="edit-input"
                    value={editForm.title}
                    onChange={(e) => handleEditFieldChange('title', e.target.value)}
                    disabled={isEditSaving}
                    autoFocus
                  />
                </label>
                <label>
                  Description
                  <textarea
                    className="edit-input edit-textarea"
                    value={editForm.description}
                    onChange={(e) => handleEditFieldChange('description', e.target.value)}
                    disabled={isEditSaving}
                  />
                </label>
                <div className="edit-modal-actions">
                  <button
                    type="button"
                    className="edit-button cancel"
                    onClick={closeEditModal}
                    disabled={isEditSaving}
                  >
                    Cancel
                  </button>
                  <button
                    type="submit"
                    className="edit-button save"
                    disabled={isEditSaving}
                  >
                    {isEditSaving ? 'Saving...' : 'Save'}
                  </button>
                </div>
              </form>
            </div>
          </div>
        )}
      </animated.div>
    </>
  );
};

export default App;
