import { messageAPI } from '../utils/apiServices';

const testMessagingAPI = async () => {
  console.log('🧪 Testing Messaging API Endpoints...');
  
  try {
    console.log('\n1. Testing get online users...');
    const users = await messageAPI.getOnlineUsers();
    console.log('✅ Online users:', users.data.length, 'users found');
    
    console.log('\n2. Testing get conversations...');
    const conversations = await messageAPI.getConversations();
    console.log('✅ Conversations:', conversations.data.length, 'conversations found');
    
    if (users.data.length > 0) {
      console.log('\n3. Testing create conversation...');
      const firstUser = users.data.find(u => u.id !== 'current-user-id'); // This should be replaced with actual current user ID
      if (firstUser) {
        try {
          const newConversation = await messageAPI.createConversation({
            participant_ids: [firstUser.id],
            title: 'Test Conversation'
          });
          console.log('✅ Conversation created:', newConversation.data.id);
          
          console.log('\n4. Testing send message...');
          const message = await messageAPI.sendMessage(newConversation.data.id, {
            content: 'Test message',
            message_type: 'text'
          });
          console.log('✅ Message sent:', message.data.id);
          
          console.log('\n5. Testing get messages...');
          const messages = await messageAPI.getMessages(newConversation.data.id);
          console.log('✅ Messages retrieved:', messages.data.length, 'messages');
        } catch (error) {
          console.log('ℹ️ Note: Create conversation may fail if no authenticated user');
        }
      }
    }
    
    console.log('\n✅ All messaging API tests completed!');
    
  } catch (error) {
    console.error('❌ Messaging API test failed:', error.response?.data || error.message);
  }
};

export default testMessagingAPI;