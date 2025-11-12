import React, { useState, useEffect, useRef } from 'react';
import { Navbar } from '../../components/Navbar/Navbar';
import { useAuth } from '../../contexts/AuthContext';
import MessageService from '../../services/MessageService';

const MessageDark = () => {
  const { user } = useAuth();
  const [conversations, setConversations] = useState([]);
  const [currentConversation, setCurrentConversation] = useState(null);
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [onlineUsers, setOnlineUsers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('conversations');
  const [error, setError] = useState(null);
  const chatAreaRef = useRef(null);

  useEffect(() => {
    if (user) {
      initializeMessaging();
    }
    return () => {
      // Remove empty conversations when leaving the page
      setConversations(prev => prev.filter(c => !!c?.last_message));
    };
  }, [user]);

  // Auto-scroll to bottom when messages or conversation changes
  useEffect(() => {
    if (chatAreaRef.current && messages.length > 0) {
      // Small delay to ensure DOM has updated
      setTimeout(() => {
        if (chatAreaRef.current) {
          chatAreaRef.current.scrollTop = chatAreaRef.current.scrollHeight;
        }
      }, 100);
    }
  }, [messages, currentConversation]);

  const initializeMessaging = async () => {
    try {
      setLoading(true);
      setError(null);
      
      // Load conversations and online users in parallel
      const [conversationsData, onlineUsersData] = await Promise.all([
        MessageService.getConversations(),
        MessageService.getOnlineUsers()
      ]);

      const safeConversationsRaw = Array.isArray(conversationsData)
        ? conversationsData
        : (conversationsData && Array.isArray(conversationsData.results) ? conversationsData.results : []);
      // On page load, exclude empty conversations returned by API
      const safeConversations = safeConversationsRaw.filter(c => !!c?.last_message);
      const safeUsers = Array.isArray(onlineUsersData) ? onlineUsersData : [];

      setConversations(safeConversations);
      setOnlineUsers(safeUsers.filter(u => u.id !== user.id)); // Exclude current user
      
      // Select first conversation if exists
      if (safeConversations.length > 0) {
        await selectConversation(safeConversations[0]);
      }
    } catch (error) {
      console.error('Failed to initialize messaging:', error);
      setError('Failed to load messaging data');
    } finally {
      setLoading(false);
    }
  };

  const selectConversation = async (conversation) => {
    try {
      setCurrentConversation(conversation);
      MessageService.setCurrentConversation(conversation);
      
      const messagesData = await MessageService.getMessages(conversation.id);
      // Ensure unique messages by id
      const uniqueById = Array.isArray(messagesData)
        ? Object.values(
            messagesData.reduce((acc, msg) => {
              if (msg && msg.id != null) acc[msg.id] = msg;
              return acc;
            }, {})
          )
        : [];
      setMessages(uniqueById);
      setActiveTab('conversations');
    } catch (error) {
      console.error('Failed to load conversation messages:', error);
      setError('Failed to load messages');
    }
  };

  const selectUser = async (selectedUser) => {
    try {
      setLoading(true);
      const conversation = await MessageService.getOrCreateConversation(selectedUser.id);
      
      // Update conversations list if it's a new conversation
      if (!conversations.find(c => c.id === conversation.id)) {
        setConversations(prev => [conversation, ...prev]);
      }
      
      await selectConversation(conversation);
    } catch (error) {
      console.error('Failed to create conversation with user:', error);
      setError('Failed to start conversation');
    } finally {
      setLoading(false);
    }
  };

  const handleSendMessage = async () => {
    if (inputMessage.trim() !== '' && currentConversation && user) {
      try {
        const newMessage = await MessageService.sendMessage(currentConversation.id, inputMessage.trim());
        // Append safely without creating duplicates (by id)
        setMessages(prev => {
          if (newMessage?.id && prev.some(m => m.id === newMessage.id)) {
            return prev;
          }
          return [...prev, newMessage];
        });
        setInputMessage('');
        
        // Update conversation in the list to show latest message
        setConversations(prev => prev.map(conv => 
          conv.id === currentConversation.id 
            ? { ...conv, last_message: newMessage, updated_at: newMessage.created_at }
            : conv
        ));
      } catch (error) {
        console.error('Failed to send message:', error);
        setError('Failed to send message');
      }
    }
  };

  return (
    <>
      <style>
        {`
        .app-container {
          display: flex;
          flex-direction: column;
          height: calc(100vh - 80px); /* Account for navbar height and margin */
          width: 100%;
          max-width: 100%;
          background-color: #1a1a1a;
          padding: 1rem;
          margin-top: 80px; /* Spacing from navbar */
          overflow-x: hidden;
          box-sizing: border-box;
        }
        .main-wrapper {
          background-color: #111827;
          border-radius: 1.5rem;
          box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3), 0 4px 6px -2px rgba(0, 0, 0, 0.2);
          display: flex;
          flex-direction: column;
          height: 100%;
          overflow: hidden;
          max-width: 100%;
          box-sizing: border-box;
          color: #e5e5e5;
        }
        .header {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 1.5rem;
          border-bottom: 1px solid #444;
          max-width: 100%;
          box-sizing: border-box;
          overflow-x: hidden;
        }
        .header-section {
          display: flex;
          align-items: center;
          flex: 1;
          gap: 1rem;
        }
        .search-input {
          width: 100%;
          padding: 0.5rem 1rem;
          border-radius: 9999px;
          border: 1px solid #444;
          outline: none;
          box-shadow: 0 0 0 2px transparent;
          transition: box-shadow 0.2s;
          background-color: #212d46;
          color: #e5e5e5;
        }
        .search-input:focus {
          box-shadow: 0 0 0 2px #3e387f;
        }
        .nav-buttons-container {
          flex: 1;
          display: flex;
          justify-content: center;
          gap: 0.5rem;
          padding-right: 25rem;
        }
        .nav-button {
          padding: 0.5rem 1.5rem;
          border-radius: 9999px;
          font-weight: 600;
          transition: background-color 0.2s;
        }
        .nav-button.active {
          background-color: #3e387f;
          color: #fff;
        }
        .nav-button.inactive {
          background-color: #444;
          color: #fff;
        }
        .nav-button.inactive:hover {
          background-color: #555;
        }
        .main-content {
          display: flex;
          flex: 1;
          overflow: hidden;
          min-height: 0;
          max-width: 100%;
        }
        .user-list {
          width: 25%;
          background-color: #111827;
          border-right: 1px solid #444;
          overflow-y: auto;
          padding: 1rem;
        }
        .user-list h2 {
          font-size: 1.125rem;
          font-weight: 700;
          margin-bottom: 1rem;
          color: #ccc;
        }
        .user-item {
          padding: 0.75rem;
          background-color: #212d46;
          border-radius: 0.75rem;
          box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.2);
          transition: background-color 0.2s;
          cursor: pointer;
          margin-bottom: 0.5rem;
          color: #e5e5e5;
        }
        .user-item:hover {
          background-color: #444;
        }
        .user-item.active {
          background-color: #3e387f;
          color: #fff;
        }
        .user-item.active:hover {
          background-color: #2c2865;
        }
        .chat-area {
          flex: 1;
          display: flex;
          flex-direction: column;
          padding: 1.5rem;
          gap: 1rem;
          overflow-y: auto;
          overflow-x: hidden;
          background-color: #111827;
          max-width: 100%;
          box-sizing: border-box;
        }
        .chat-message {
          max-width: 28rem;
          padding: 1rem;
          border-radius: 0.75rem;
          box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.2), 0 1px 2px 0 rgba(0, 0, 0, 0.12);
        }
        .chat-message.user {
          background-color: #3e387f;
          color: #fff;
          align-self: flex-end;
        }
        .chat-message.other {
          background-color: #444;
          color: #e5e5e5;
          align-self: flex-start;
        }
        .empty-chat {
          flex: 1;
          display: flex;
          align-items: center;
          justify-content: center;
          color: #666;
          font-size: 1.125rem;
          font-weight: 300;
        }
        .message-input-container {
          padding: 1.5rem;
          background-color: #111827;
          border-top: 1px solid #444;
          max-width: 100%;
          box-sizing: border-box;
        }
        .message-input-container div {
          display: flex;
          align-items: center;
          gap: 1rem;
        }
        .message-input {
          flex: 1;
          padding: 1rem 1.5rem;
          border-radius: 9999px;
          border: 2px solid #444;
          outline: none;
          transition: box-shadow 0.2s;
          background-color: #212d46;
          color: #e5e5e5;
        }
        .message-input:focus {
          box-shadow: 0 0 0 2px #3e387f;
        }
          input[type="text"], input[type="url"], textarea {
    width: 80%;
    padding: 10px;
    border: none;
    /* height: 2rem; */
    border-radius: 3px;
    background-color: #182134ff;
    transition: 0.4s all ease-in-outx;
}
        .send-button {
          padding: 1rem;
          border-radius: 9999px;
          background-color: #3e387f;
          color: #fff;
          transition: background-color 0.2s;
          box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2), 0 2px 4px -1px rgba(0, 0, 0, 0.12);
          display: flex;
          align-items: center;
          justify-content: center;
          cursor: pointer;
        }
        .send-button:hover {
          background-color: #2c2865;
        }
        .send-icon {
          width: 1.5rem;
          height: 1.5rem;
        }
        .footer {
          display: flex;
          justify-content: center;
          align-items: center;
          padding: 1rem;
          font-size: 0.75rem;
          color: #888;
          gap: 1.5rem;
          border-top: 1px solid #444;
          background-color: #1a1a1a;
        }
        .footer a {
          transition: color 0.2s;
        }
        .footer a:hover {
          color: #3e387f;
        }
        .social-link {
          display: flex;
          align-items: center;
          gap: 0.25rem;
        }
        `}
      </style>
      <Navbar />
      
      <div className="app-container">
        <div className="main-wrapper">
          {/* Header */}
          <header className="header">
            <div className="header-section">
              <input
                type="text"
                placeholder="Search..."
                className="search-input"
              />
            </div>
            <div className="nav-buttons-container">
              <button 
                className={`nav-button ${activeTab === 'conversations' ? 'active' : 'inactive'}`}
                onClick={() => setActiveTab('conversations')}
              >
                Conversations
              </button>
              <button 
                className={`nav-button ${activeTab === 'people' ? 'active' : 'inactive'}`}
                onClick={() => setActiveTab('people')}
              >
                People
              </button>
            </div>
          </header>

          {/* Main Content: User List and Chat Area */}
          <main className="main-content">
            {/* User/Conversation List */}
            <div className="user-list">
              {loading ? (
                <div>Loading...</div>
              ) : error ? (
                <div style={{ color: 'red' }}>Error: {error}</div>
              ) : activeTab === 'conversations' ? (
                <div>
                  <h2>Conversations</h2>
                  {!Array.isArray(conversations) || conversations.length === 0 ? (
                    <div className="user-item">No conversations yet</div>
                  ) : (
                    conversations.map(conversation => {
                      const otherUser = MessageService.getOtherParticipant(conversation, user?.id);
                      const title = MessageService.getConversationTitle(conversation, user?.id);
                      return (
                        <div 
                          key={conversation.id} 
                          className={`user-item ${currentConversation?.id === conversation.id ? 'active' : ''}`}
                          onClick={() => selectConversation(conversation)}
                        >
                          <div style={{ fontWeight: 'bold' }}>{title}</div>
                          {conversation.last_message && (
                            <div style={{ fontSize: '0.8em', color: '#888', marginTop: '4px' }}>
                              {conversation.last_message.content.substring(0, 50)}{conversation.last_message.content.length > 50 ? '...' : ''}
                            </div>
                          )}
                        </div>
                      );
                    })
                  )}
                </div>
              ) : (
                <div>
                  <h2>Online Users</h2>
                  {onlineUsers.length === 0 ? (
                    <div className="user-item">No users online</div>
                  ) : (
                    onlineUsers.map(onlineUser => (
                      <div 
                        key={onlineUser.id} 
                        className="user-item"
                        onClick={() => selectUser(onlineUser)}
                      >
                        <div style={{ fontWeight: 'bold' }}>{onlineUser.username}</div>
                        <div style={{ fontSize: '0.8em', color: '#888' }}>{onlineUser.email}</div>
                      </div>
                    ))
                  )}
                </div>
              )}
            </div>

            {/* Chat Area */}
            <div className="chat-area" ref={chatAreaRef}>
              {!currentConversation ? (
                <div className="empty-chat">
                  {activeTab === 'conversations' ? 'Select a conversation to start messaging' : 'Select a user to start messaging'}
                </div>
              ) : messages.length > 0 ? (
                messages.map((msg, index) => {
                  const isCurrentUser = msg.sender.id === user?.id;
                  return (
                    <div
                      key={msg.id || index}
                      className={`chat-message ${isCurrentUser ? 'user' : 'other'}`}
                    >
                      {!isCurrentUser && (
                        <div style={{ fontSize: '0.8em', color: '#aaa', marginBottom: '4px' }}>
                          {msg.sender.username}
                        </div>
                      )}
                      <div>{msg.content}</div>
                      <div style={{ fontSize: '0.7em', color: '#888', marginTop: '4px' }}>
                        {new Date(msg.created_at).toLocaleTimeString()}
                      </div>
                    </div>
                  );
                })
              ) : (
                <div className="empty-chat">
                  No messages yet. Start the conversation!
                </div>
              )}
            </div>
          </main>

          {/* Message Input */}
          <div className="message-input-container">
            <div>
              <input
                type="text"
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                onKeyPress={(e) => {
                  if (e.key === 'Enter') {
                    handleSendMessage();
                  }
                }}
                placeholder={currentConversation ? "Type a message..." : "Select a conversation to start messaging"}
                className="message-input"
                disabled={!currentConversation || loading}
              />
              <button
                onClick={handleSendMessage}
                className="send-button"
                disabled={!currentConversation || loading || !inputMessage.trim()}
              >
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="send-icon">
                  <path d="M3.478 2.405a.75.75 0 00-.926.94l2.432 7.917H12.75a.75.75 0 010 1.5H4.984l-2.432 7.918a.75.75 0 00.926.94 60.519 60.519 0 0018.445-8.986.75.75 0 000-1.218A60.517 60.517 0 003.478 2.405z" />
                </svg>
              </button>
            </div>
          </div>

          {/* Footer */}
          {/*<footer className="footer">
            <a href="#">Privacy Policy</a>
            <a href="#">Terms & Conditions</a>
            <div className="social-link">
              <span role="img" aria-label="discord">
                👾
              </span>
              <span>Discord</span>
            </div>
            <div className="social-link">
              <span role="img" aria-label="instagram">
                📸
              </span>
              <span>Instagram</span>
            </div>
          </footer>*/}
        </div>
      </div>
    </>
  );
};

export default MessageDark;
