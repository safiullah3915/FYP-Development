import React, { useState, useMemo, useCallback } from 'react';
import { useSpring, animated } from 'react-spring';

// This is the main component that renders the settings page.
const AccountSettingsLight = () => {
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
  const [editingItem, setEditingItem] = useState(null);
  const [editForm, setEditForm] = useState({ title: '', description: '' });

  const fade = useSpring({
    from: { opacity: 0 },
    to: { opacity: 1 },
    config: { duration: 500 },
  });

  // Handle the toggle for making the profile public.
  const handleToggle = () => {
    setIsProfilePublic(prev => !prev);
  };

  const handleNewItemChange = (type, field, value) => {
    const updater = type === 'references'
      ? setNewReference
      : type === 'skills'
        ? setNewSkill
        : setNewExperience;

    updater(prev => ({ ...prev, [field]: value }));
  };

  const handleAddItem = (type) => {
    const targetState =
      type === 'references'
        ? { setter: setReferences, values: newReference, reset: () => setNewReference({ title: '', description: '' }) }
        : type === 'skills'
          ? { setter: setSkills, values: newSkill, reset: () => setNewSkill({ title: '', description: '' }) }
          : { setter: setExperiences, values: newExperience, reset: () => setNewExperience({ title: '', description: '' }) };

    const { title, description } = targetState.values;
    if (!title.trim() || !description.trim()) {
      alert('Please provide both name and description.');
      return;
    }

    targetState.setter(prev => {
      const updated = [...prev, { title: title.trim(), description: description.trim() }];
      const totalPages = Math.ceil(updated.length / PAGE_SIZE);
      if (type === 'references') setReferencesPage(totalPages);
      else if (type === 'skills') setSkillsPage(totalPages);
      else setExperiencesPage(totalPages);
      return updated;
    });
    targetState.reset();
  };

  const handleRemoveItem = (type, index) => {
    if (!window.confirm('Are you sure you want to delete this entry?')) {
      return;
    }

    const setter = type === 'references' ? setReferences : type === 'skills' ? setSkills : setExperiences;
    setter(prev => {
      const updated = prev.filter((_, i) => i !== index);
      const totalPages = Math.max(1, Math.ceil(updated.length / PAGE_SIZE));
      if (type === 'references') setReferencesPage(prevPage => Math.min(prevPage, totalPages));
      else if (type === 'skills') setSkillsPage(prevPage => Math.min(prevPage, totalPages));
      else setExperiencesPage(prevPage => Math.min(prevPage, totalPages));
      return updated;
    });
  };

  const handleEditItem = (type, index) => {
    const list = type === 'references' ? references : type === 'skills' ? skills : experiences;
    const item = list[index];
    if (!item) return;

    setEditingItem({ type, index });
    setEditForm({
      title: item.title || '',
      description: item.description || ''
    });
  };

  const handleEditFieldChange = (field, value) => {
    setEditForm(prev => ({ ...prev, [field]: value }));
  };

  const closeEditModal = () => {
    setEditingItem(null);
    setEditForm({ title: '', description: '' });
  };

  const handleSaveEdit = (event) => {
    event.preventDefault();
    if (!editingItem) return;

    const title = editForm.title.trim();
    const description = editForm.description.trim();

    if (!title || !description) {
      alert('Both fields are required.');
      return;
    }

    const setter = editingItem.type === 'references'
      ? setReferences
      : editingItem.type === 'skills'
        ? setSkills
        : setExperiences;

    setter(prev =>
      prev.map((entry, i) =>
        i === editingItem.index ? { title, description } : entry
      )
    );

    closeEditModal();
  };

  const PAGE_SIZE = 3;

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

  const handleSave = () => {
    // You can implement the save logic here.
    console.log('Saving settings:', { isProfilePublic, references, skills, experiences });
  };

  return (
    <>
      <style>
        {`
        body {
          background-color: #F9FAFB;
          color: #1F2937;
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
          background-color: #FFFFFF;
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
          background-color: #F9FAFB;
          border-radius: 1rem;
          padding: 1.5rem;
          position: relative;
          display: flex;
          flex-direction: column;
          align-items: center;
          border: 1px solid #E5E7EB;
        }
        @media (min-width: 1024px) {
          .inner-card {
            padding: 2.5rem;
          }
        }
        .public-profile-section {
          width: 100%;
          border-bottom: 1px solid #E5E7EB;
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
          background-color: #D1D5DB;
          transition: .4s;
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
          background-color: #3e387f;
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
          color: #3e387f;
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
          color: #6B7280;
        }
        .region-checkbox {
          width: 1rem;
          height: 1rem;
          border-radius: 0.25rem;
          border: 1px solid #9CA3AF;
          background-color: #F9FAFB;
          transition: background-color 0.15s;
        }
        .region-checkbox:checked {
          background-color: #3e387f;
          border-color: #3e387f;
        }
        .region-checkbox:focus {
          outline: none;
          box-shadow: 0 0 0 2px #3e387f;
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
          border: 2px solid #3e387f;
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
          background-color: rgba(255, 255, 255, 0.5);
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
          color:#1F2937;
          margin: 0;
        }
        .logout-button {
          display: none;
          align-items: center;
          color: #3e387f;
          transition: color 0.15s;
        }
        .logout-button:hover {
          color: #2e285f;
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
          color: #6B7280;
          font-size: 0.875rem;
          line-height: 1.25rem;
          margin: 0;
        }
        .profile-description {
          margin-top: 1rem;
          font-size: 0.875rem;
          line-height: 1.625;
          color: #4B5563;
        }
        .profile-description p {
          font-weight: 600;
          margin: 0;
        }
        .section-divider {
          width: 100%;
          margin-top: 2rem;
          border-top: 1px solid #E5E7EB;
          padding-top: 2rem;
        }
        .section-title {
          font-size: 1.25rem;
          line-height: 1.75rem;
          font-weight: 700;
          margin-bottom: 1rem;
        }
        .collection-form {
          background-color: #F9FAFB;
          border-radius: 1rem;
          padding: 1.5rem;
          border: 1px solid #E5E7EB;
          display: flex;
          flex-direction: column;
          gap: 1rem;
          margin-bottom: 1rem;
        }
        .collection-form label {
          font-size: 0.9rem;
          color: #4B5563;
          font-weight: 600;
          display: flex;
          flex-direction: column;
          gap: 0.35rem;
        }
        .collection-input {
          background-color: #FFFFFF;
          color: #111827;
          border: 1px solid #D1D5DB;
          border-radius: 0.65rem;
          padding: 0.65rem 0.75rem;
          font-size: 0.95rem;
          transition: border-color 0.2s ease, box-shadow 0.2s ease;
        }
        .collection-input:focus {
          outline: none;
          border-color: #3e387f;
          box-shadow: 0 0 0 3px rgba(62, 56, 127, 0.15);
        }
        .collection-actions {
          display: flex;
          justify-content: flex-end;
        }
        .add-button {
          background-color: #3e387f;
          color: white;
          border: none;
          padding: 0.6rem 1.4rem;
          border-radius: 999px;
          font-weight: 600;
          cursor: pointer;
          transition: background-color 0.2s ease, transform 0.2s ease;
        }
        .add-button:hover {
          background-color: #2e285f;
          transform: translateY(-1px);
        }
        .data-table {
          width: 100%;
          border-collapse: collapse;
          border-radius: 1rem;
          overflow: hidden;
          border: 1px solid #E5E7EB;
          background-color: #FFFFFF;
        }
        .data-table th,
        .data-table td {
          padding: 0.85rem 1rem;
          border-bottom: 1px solid #E5E7EB;
          text-align: left;
          font-size: 0.92rem;
          color: #374151;
        }
        .data-table thead {
          background-color: #F3F4F6;
        }
        .data-table th {
          font-weight: 700;
          color: #1F2937;
        }
        .data-table tbody tr:last-child td {
          border-bottom: none;
        }
        .table-actions {
          display: flex;
          gap: 0.6rem;
        }
        .table-button {
          padding: 0.45rem 0.9rem;
          border-radius: 999px;
          font-weight: 600;
          border: none;
          cursor: pointer;
          transition: opacity 0.2s ease, transform 0.2s ease;
        }
        .table-button:hover {
          opacity: 0.85;
          transform: translateY(-1px);
        }
        .table-button.edit {
          background-color: #2563EB;
          color: #FFFFFF;
        }
        .table-button.delete {
          background-color: #EF4444;
          color: #FFFFFF;
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
          border: 1px solid #D1D5DB;
          background-color: #FFFFFF;
          color: #1F2937;
          font-size: 1.1rem;
          font-weight: 600;
          cursor: pointer;
          transition: background-color 0.2s ease, color 0.2s ease, opacity 0.2s ease;
          display: flex;
          align-items: center;
          justify-content: center;
          line-height: 1;
        }
        .pagination-button:hover:not(:disabled) {
          background-color: #3e387f;
          color: #FFFFFF;
        }
        .pagination-button:disabled {
          opacity: 0.4;
          cursor: not-allowed;
        }
        .pagination-status {
          color: #1F2937;
          font-weight: 600;
          letter-spacing: 0.05em;
        }
        .edit-modal-overlay {
          position: fixed;
          inset: 0;
          background: rgba(15, 23, 42, 0.35);
          display: flex;
          align-items: center;
          justify-content: center;
          padding: 1.5rem;
          z-index: 60;
          animation: tableFade 0.25s ease;
        }
        .edit-modal {
          background-color: #FFFFFF;
          border: 1px solid #E5E7EB;
          border-radius: 1rem;
          padding: 1.5rem 1rem;
          width: 100%;
          max-width: 360px;
          box-shadow: 0 25px 50px -12px rgba(15, 23, 42, 0.35);
        }
        .edit-modal h4 {
          margin: 0 0 1rem 0;
          color: #1F2937;
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
          color: #4B5563;
          font-size: 0.9rem;
          font-weight: 600;
        }
        .edit-input {
          background-color: #F9FAFB;
          border: 1px solid #D1D5DB;
          border-radius: 0.6rem;
          padding: 0.6rem 0.75rem;
          color: #111827;
          font-size: 0.95rem;
          transition: border-color 0.2s ease, box-shadow 0.2s ease;
        }
        .edit-input[type="text"] {
          background-color: #F9FAFB;
        }
        .edit-input:focus {
          outline: none;
          border-color: #3e387f;
          box-shadow: 0 0 0 3px rgba(62, 56, 127, 0.15);
        }
        .edit-textarea {
          min-height: 100px;
          resize: vertical;
          background-color: #F9FAFB;
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
          transition: background-color 0.2s ease, transform 0.2s ease, opacity 0.2s ease;
        }
        .edit-button:hover {
          transform: translateY(-1px);
        }
        .edit-button.cancel {
          background-color: transparent;
          border: 1px solid #D1D5DB;
          color: #1F2937;
        }
        .edit-button.save {
          background-color: #3e387f;
          color: #FFFFFF;
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
        .save-button {
          margin-top: 2.5rem;
          width: 100%;
          padding-left: 3rem;
          padding-right: 3rem;
          padding-top: 0.75rem;
          padding-bottom: 0.75rem;
          background-color: #3e387f;
          color: white;
          font-weight: 600;
          border-radius: 9999px;
          box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
          transition: all 0.15s;
          border: none;
          cursor: pointer;
        }
        .save-button:hover {
          background-color: #2e285f;
          transform: scale(1.05);
        }
        @media (min-width: 1024px) {
          .save-button {
            width: auto;
          }
        }
        `}
      </style>
      <animated.div style={fade} className="container">
        <div className="main-card">
          {/* Main content area */}
          <div className="content-area">
            <div className="inner-card">
              {/* User Profile Section */}
              <div className="user-profile">
                <div className="profile-picture">
                  <img
                    src="https://placehold.co/128x128/3e387f/fff?text=SAFI"
                    alt="Profile"
                  />
                  <div className="edit-overlay">
                    <span className="text-sm">Edit</span>
                  </div>
                </div>
                <div className="profile-info">
                  <div className="profile-header">
                    <h2>Safi Ullah</h2>
                    <button className="logout-button" onClick={() => console.log('Logged out.')}>
                      <span>Logout</span>
                      <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1m0-16V6m6 10v-3a4 4 0 00-4-4H7a4 4 0 00-4 4v3" />
                      </svg>
                    </button>
                  </div>
                  <p>Asia, Pakistan</p>
                  <div className="profile-description">
                    <p>tldr: Safi from Pakistan is a skilled programmer with over 2 years of experience building backend solutions with Python.</p>
                  </div>
                </div>
              </div>

              {/* Make Profile Public Section */}
              <div className="section-divider">
                <div className="public-profile-header">
                  <h3 className="public-profile-title">Make profile public</h3>
                  <div className="relative">
                    <span className={`toggle-text ${isProfilePublic ? 'on' : ''}`}>
                      {isProfilePublic ? 'On' : 'Off'}
                    </span>
                    <label className="toggle-switch">
                      <input type="checkbox" checked={isProfilePublic} onChange={handleToggle} />
                      <span className="slider"></span>
                    </label>
                  </div>
                </div>
                <p className="text-gray-600 text-sm mt-2">
                  Make your profile discoverable to other members on the platform.
                </p>
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
                      placeholder="Brief description of this skill"
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
              
              {/* Save Button */}
              <button
                onClick={handleSave}
                className="save-button"
              >
                Save & Exit
              </button>

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
                          autoFocus
                        />
                      </label>
                      <label>
                        Description
                        <textarea
                          className="edit-input edit-textarea"
                          value={editForm.description}
                          onChange={(e) => handleEditFieldChange('description', e.target.value)}
                        />
                      </label>
                      <div className="edit-modal-actions">
                        <button
                          type="button"
                          className="edit-button cancel"
                          onClick={closeEditModal}
                        >
                          Cancel
                        </button>
                        <button
                          type="submit"
                          className="edit-button save"
                        >
                          Save
                        </button>
                      </div>
                    </form>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </animated.div>
    </>
  );
};

export default App;
