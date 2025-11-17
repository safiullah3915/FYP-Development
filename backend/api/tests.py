from django.test import TestCase
from django.contrib.auth import get_user_model
from rest_framework.test import APITestCase, APIClient
from rest_framework import status
from django.urls import reverse
import json
import bcrypt
from .models import Startup, StartupTag, Position, Application
from .messaging_models import Conversation
from .recommendation_models import UserOnboardingPreferences
from .serializers import UserOnboardingPreferencesSerializer

User = get_user_model()


class BcryptUserMixin:
    """Test helper mixin to create users with bcrypt-hashed passwords"""

    def create_bcrypt_user(self, username='testuser', email='test@example.com', password='testpassword123', **extra):
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        defaults = {
            'username': username,
            'email': email,
            'password': hashed_password,
        }
        defaults.update(extra)
        return User.objects.create(**defaults)


class AuthenticationTestCase(BcryptUserMixin, APITestCase):
    """Test cases for authentication endpoints"""
    
    def setUp(self):
        self.signup_url = reverse('signup')
        self.login_url = reverse('login')
        self.user_data = {
            'username': 'testuser',
            'email': 'test@example.com',
            'password': 'testpassword123'
        }
    
    def test_user_signup(self):
        """Test user registration"""
        response = self.client.post(self.signup_url, self.user_data)
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertIn('message', response.data)
        self.assertIn('user', response.data)
        self.assertTrue(User.objects.filter(email=self.user_data['email']).exists())
    
    def test_user_login(self):
        """Test user login"""
        # Create user first with bcrypt password
        self.create_bcrypt_user(
            username=self.user_data['username'],
            email=self.user_data['email'],
            password=self.user_data['password']
        )
        
        login_data = {
            'email': self.user_data['email'],
            'password': self.user_data['password']
        }
        response = self.client.post(self.login_url, login_data)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('message', response.data)
        self.assertIn('user', response.data)
    
    def test_invalid_login(self):
        """Test login with invalid credentials"""
        login_data = {
            'email': 'nonexistent@example.com',
            'password': 'wrongpassword'
        }
        response = self.client.post(self.login_url, login_data)
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)


class StartupTestCase(BcryptUserMixin, APITestCase):
    """Test cases for startup endpoints"""
    
    def setUp(self):
        self.user = self.create_bcrypt_user(
            username='testuser',
            email='test@example.com',
            password='testpassword123'
        )
        self.client.force_login(self.user)
        
        self.startup_data = {
            'title': 'Test Startup',
            'description': 'This is a test startup description that is long enough',
            'field': 'Technology',
            'type': 'marketplace',
            'category': 'saas'
        }
    
    def test_create_startup(self):
        """Test creating a startup"""
        url = reverse('startup_create')
        response = self.client.post(url, self.startup_data)
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertTrue(Startup.objects.filter(title=self.startup_data['title']).exists())
    
    def test_marketplace_list(self):
        """Test marketplace listing"""
        # Create a startup
        startup = Startup.objects.create(
            owner=self.user,
            title='Test Marketplace Startup',
            description='Test description for marketplace',
            field='Technology',
            type='marketplace'
        )
        
        url = reverse('marketplace_list')
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('results', response.data)
    
    def test_startup_detail(self):
        """Test startup detail view"""
        startup = Startup.objects.create(
            owner=self.user,
            title='Test Startup Detail',
            description='Test description for detail view',
            field='Technology'
        )
        
        url = reverse('startup_detail', kwargs={'pk': startup.id})
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['title'], startup.title)


class ApplicationTestCase(BcryptUserMixin, APITestCase):
    """Test cases for application endpoints"""
    
    def setUp(self):
        self.user = self.create_bcrypt_user(
            username='testuser',
            email='test@example.com',
            password='testpassword123'
        )
        self.client.force_login(self.user)
        
        # Create a startup and position
        self.startup = Startup.objects.create(
            owner=self.user,
            title='Test Startup',
            description='Test description for applications',
            field='Technology',
            type='collaboration'
        )
        
        self.position = Position.objects.create(
            startup=self.startup,
            title='Test Position',
            description='Test position description'
        )
    
    def test_apply_for_collaboration(self):
        """Test applying for a collaboration"""
        url = reverse('apply_collaboration', kwargs={'pk': self.startup.id})
        application_data = {
            'position_id': str(self.position.id),
            'cover_letter': 'This is my cover letter',
            'experience': 'I have 5 years of experience',
            'resume_url': 'https://example.com/resume.pdf'
        }
        
        response = self.client.post(url, application_data)
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertIn('message', response.data)
        self.assertTrue(Application.objects.filter(
            applicant=self.user,
            position=self.position
        ).exists())
    
    def test_user_applications(self):
        """Test getting user applications"""
        # Create an application
        Application.objects.create(
            startup=self.startup,
            position=self.position,
            applicant=self.user,
            cover_letter='Test cover letter'
        )
        
        url = reverse('user_applications')
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('results', response.data)


class SearchTestCase(BcryptUserMixin, APITestCase):
    """Test cases for search functionality"""
    
    def setUp(self):
        self.user = self.create_bcrypt_user(
            username='testuser',
            email='test@example.com',
            password='testpassword123'
        )
        
        # Create test startups
        Startup.objects.create(
            owner=self.user,
            title='AI Startup',
            description='Artificial intelligence startup',
            field='Technology',
            type='marketplace'
        )
        
        Startup.objects.create(
            owner=self.user,
            title='E-commerce Platform',
            description='Online shopping platform',
            field='E-commerce',
            type='collaboration'
        )
    
    def test_search_startups(self):
        """Test searching startups"""
        url = reverse('search')
        response = self.client.get(url, {'q': 'AI'})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('results', response.data)


class OnboardingPreferencesSerializerTest(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            username='investor',
            email='investor@example.com',
            password='complexpass123',
            role='investor'
        )
        self.prefs = UserOnboardingPreferences.objects.create(user=self.user)
    
    def test_investor_payload_persists_and_maps(self):
        payload = {
            'sectors': ['AI', 'Climate'],
            'stages': ['Seed'],
            'round_types': ['Equity'],
            'instruments': ['SAFE'],
            'business_models': ['B2B SaaS'],
            'geographies': ['US'],
            'check_size': {'min': '100k', 'max': '500k', 'currency': 'usd'},
            'target_ownership': {'min_pct': 5, 'max_pct': 10},
            'valuation_caps': {'post_money_max': '50M'},
            'traction': {'arr_min': '1M'},
            'support_preferences': ['Go-to-market'],
            'collaboration_style': 'hands-on',
            'lead_preference': 'lead',
            'decision_speed': 'fast',
        }
        serializer = UserOnboardingPreferencesSerializer(
            instance=self.prefs,
            data={'investor_preferences': payload, 'onboarding_completed': True},
            partial=True
        )
        self.assertTrue(serializer.is_valid(), serializer.errors)
        serializer.save()
        
        self.prefs.refresh_from_db()
        investor_profile = self.prefs.investor_profile
        self.assertEqual(investor_profile['sectors'], ['ai', 'climate'])
        self.assertIn('ai', self.prefs.selected_categories)
        self.assertIn('seed', self.prefs.preferred_startup_stages)
        self.assertIn('round:equity', self.prefs.preferred_engagement_types)
        self.assertIn('instr:safe', self.prefs.preferred_engagement_types)
        self.assertIn('geo:us', self.prefs.selected_tags)
        self.assertIn('check_min:100k', self.prefs.selected_tags)
        self.assertIn('leadpref:lead', self.prefs.selected_tags)
        self.assertTrue(self.prefs.onboarding_completed)
        
        # Representation should include investor_profile
        data = UserOnboardingPreferencesSerializer(instance=self.prefs).data
        self.assertIn('investor_profile', data)
        self.assertEqual(data['investor_profile']['decision_speed'], 'fast')
        self.assertIn('count', response.data)
    
    def test_search_with_filters(self):
        """Test searching with type filter"""
        url = reverse('search')
        response = self.client.get(url, {'q': 'startup', 'type': 'marketplace'})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('results', response.data)


class MessagingTestCase(BcryptUserMixin, APITestCase):
    """Test cases for messaging endpoints"""

    def setUp(self):
        self.user = self.create_bcrypt_user(
            username='messaging-user',
            email='messaging@example.com',
            password='securepassword123'
        )
        self.other_user = self.create_bcrypt_user(
            username='other-user',
            email='other@example.com',
            password='securepassword123'
        )
        self.conversations_url = reverse('conversations_list')
        self.client.force_login(self.user)

    def test_create_conversation_includes_current_user(self):
        payload = {
            'participant_ids': [str(self.other_user.id)],
            'title': 'Test Chat'
        }
        response = self.client.post(self.conversations_url, payload, format='json')
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertIn('id', response.data)
        participant_ids = {participant['id'] for participant in response.data['participants']}
        self.assertSetEqual(participant_ids, {str(self.user.id), str(self.other_user.id)})

    def test_send_message_returns_full_payload(self):
        conversation = Conversation.objects.create(title='Existing Chat')
        conversation.participants.set([self.user, self.other_user])
        messages_url = reverse('messages_list', kwargs={'conversation_id': conversation.id})
        payload = {
            'content': 'Hello there!',
            'message_type': 'text'
        }
        response = self.client.post(messages_url, payload, format='json')
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertIn('id', response.data)
        self.assertEqual(response.data['content'], payload['content'])
        self.assertEqual(response.data['sender']['id'], str(self.user.id))


class MessagingAPITestCase(BcryptUserMixin, APITestCase):
    """End-to-end messaging flow tests using real authentication"""

    def setUp(self):
        self.password = 'securepassword123'
        self.user_one = self.create_bcrypt_user(
            username='api-user-one',
            email='api-user-one@example.com',
            password=self.password
        )
        self.user_two = self.create_bcrypt_user(
            username='api-user-two',
            email='api-user-two@example.com',
            password=self.password
        )
        self.login_url = reverse('login')
        self.conversations_url = reverse('conversations_list')

    def authenticate(self, client, email, password):
        response = client.post(self.login_url, {
            'email': email,
            'password': password
        }, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        token = response.data.get('auth_token')
        self.assertIsNotNone(token)
        client.credentials(HTTP_AUTHORIZATION=f'Bearer {token}')
        return response

    def test_conversation_listing_requires_authentication(self):
        unauthenticated_client = APIClient()
        response = unauthenticated_client.get(self.conversations_url)
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)
        self.assertIn('error', response.data)

    def test_full_messaging_flow_between_two_users(self):
        client_one = APIClient()
        self.authenticate(client_one, self.user_one.email, self.password)

        create_response = client_one.post(self.conversations_url, {
            'participant_ids': [str(self.user_two.id)]
        }, format='json')
        self.assertEqual(create_response.status_code, status.HTTP_201_CREATED)
        conversation_id = create_response.data['id']

        messages_url = reverse('messages_list', kwargs={'conversation_id': conversation_id})
        send_response = client_one.post(messages_url, {
            'content': 'Hello from user one!',
            'message_type': 'text'
        }, format='json')
        self.assertEqual(send_response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(send_response.data['sender']['id'], str(self.user_one.id))

        client_two = APIClient()
        self.authenticate(client_two, self.user_two.email, self.password)
        list_response = client_two.get(self.conversations_url)
        self.assertEqual(list_response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(list_response.data), 1)
        self.assertEqual(list_response.data[0]['id'], conversation_id)

        messages_response = client_two.get(messages_url)
        self.assertEqual(messages_response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(messages_response.data), 1)
        self.assertEqual(messages_response.data[0]['content'], 'Hello from user one!')
        self.assertEqual(messages_response.data[0]['sender']['id'], str(self.user_one.id))
