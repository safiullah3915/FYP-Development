import { messageAPI } from '../utils/apiServices';

class MessageService {
  constructor() {
    this.conversations = [];
    this.currentConversation = null;
    this.messages = {};
    this.onlineUsers = [];
  }

  async getOnlineUsers() {
    try {
      const response = await messageAPI.getOnlineUsers();
      this.onlineUsers = response.data;
      return this.onlineUsers;
    } catch (error) {
      console.error('Failed to get online users:', error);
      throw error;
    }
  }

  async getConversations() {
    try {
      const response = await messageAPI.getConversations();
      // Normalize to a plain array (supports paginated and non-paginated)
      const data = response?.data;
      const list = Array.isArray(data)
        ? data
        : (data && Array.isArray(data.results) ? data.results : []);
      this.conversations = list;
      return list;
    } catch (error) {
      console.error('Failed to get conversations:', error);
      throw error;
    }
  }

  async getConversation(conversationId) {
    try {
      const response = await messageAPI.getConversation(conversationId);
      return response.data;
    } catch (error) {
      console.error('Failed to get conversation:', error);
      throw error;
    }
  }

  async createConversation(participantIds, title = '') {
    try {
      const response = await messageAPI.createConversation({
        participant_ids: participantIds,
        title
      });
      const newConversation = response.data;
      this.conversations.unshift(newConversation);
      return newConversation;
    } catch (error) {
      console.error('Failed to create conversation:', error);
      throw error;
    }
  }

  async getMessages(conversationId) {
    try {
      const response = await messageAPI.getMessages(conversationId);
      this.messages[conversationId] = response.data;
      return response.data;
    } catch (error) {
      console.error('Failed to get messages:', error);
      throw error;
    }
  }

  async sendMessage(conversationId, content) {
    try {
      const response = await messageAPI.sendMessage(conversationId, {
        content,
        message_type: 'text'
      });
      const newMessage = response.data;
      
      // Add message to local cache
      if (!this.messages[conversationId]) {
        this.messages[conversationId] = [];
      }
      this.messages[conversationId].push(newMessage);

      // Update conversation's last message and updated_at
      const conversation = this.conversations.find(c => c.id === conversationId);
      if (conversation) {
        conversation.last_message = newMessage;
        conversation.updated_at = newMessage.created_at;
      }

      return newMessage;
    } catch (error) {
      console.error('Failed to send message:', error);
      throw error;
    }
  }

  async getOrCreateConversation(userId) {
    try {
      // Check if conversation with this user already exists
      const existingConversation = this.conversations.find(conv => 
        conv.participants.some(p => p.id === userId) && conv.participants.length === 2
      );

      if (existingConversation) {
        return existingConversation;
      }

      // Create new conversation
      const newConversation = await this.createConversation([userId]);
      return newConversation;
    } catch (error) {
      console.error('Failed to get or create conversation:', error);
      throw error;
    }
  }

  setCurrentConversation(conversation) {
    this.currentConversation = conversation;
  }

  getCurrentConversation() {
    return this.currentConversation;
  }

  getConversationMessages(conversationId) {
    return this.messages[conversationId] || [];
  }

  // Helper method to get other participant in a conversation
  getOtherParticipant(conversation, currentUserId) {
    return conversation.participants.find(p => p.id !== currentUserId);
  }

  // Format conversation title for display
  getConversationTitle(conversation, currentUserId) {
    if (conversation.title) {
      return conversation.title;
    }
    
    if (conversation.participants.length === 2) {
      const otherUser = this.getOtherParticipant(conversation, currentUserId);
      return otherUser ? otherUser.username : 'Unknown User';
    }
    
    return `Group (${conversation.participants.length} members)`;
  }
}

// Export singleton instance
export default new MessageService();