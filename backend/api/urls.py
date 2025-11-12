from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

# URL patterns for the API
urlpatterns = [
	# Home endpoint
	path('', views.home, name='home'),
	
	# Authentication endpoints
	path('signup', views.SignupView.as_view(), name='signup'),
	path('auth/login', views.LoginView.as_view(), name='login'),
	path('auth/test', views.auth_test, name='auth_test'),
	path('auth/cookie-test', views.cookie_test, name='cookie_test'),
	path('auth/refresh', views.RefreshTokenView.as_view(), name='refresh_token'),
	path('auth/forget-password', views.forget_password, name='forget_password'),
	path('auth/send-verification-code', views.SendVerificationCodeView.as_view(), name='send_verification_code'),
	path('auth/verify', views.VerifyEmailView.as_view(), name='verify_email'),
	path('auth/logout', views.logout, name='logout'),
	
	# Startup management endpoints
	path('api/startups', views.StartupCreateView.as_view(), name='startup_create'),
	path('api/startups/<uuid:pk>', views.StartupDetailView.as_view(), name='startup_detail'),
	path('api/marketplace', views.MarketplaceListView.as_view(), name='marketplace_list'),
	path('api/collaborations', views.CollaborationListView.as_view(), name='collaboration_list'),
	
	# Collaboration endpoints
	path('api/collaborations/<uuid:pk>/apply', views.ApplyForCollaborationView.as_view(), name='apply_collaboration'),
	path('api/users/applications', views.UserApplicationsView.as_view(), name='user_applications'),
	# Entrepreneur application management
	path('api/startups/<uuid:pk>/applications', views.StartupApplicationsView.as_view(), name='startup_applications'),
	path('api/applications/<uuid:pk>/approve', views.ApproveApplicationView.as_view(), name='approve_application'),
	path('api/applications/<uuid:pk>/decline', views.DeclineApplicationView.as_view(), name='decline_application'),

	# UC5: Positions management
	path('api/positions', views.AllPositionsView.as_view(), name='all_positions'),  # List all available positions
	path('api/startups/<uuid:pk>/positions', views.StartupPositionsView.as_view(), name='startup_positions'),
	path('api/positions/<uuid:pk>', views.PositionDetailView.as_view(), name='position_detail'),
	path('api/positions/<uuid:pk>/applications', views.PositionApplicationsView.as_view(), name='position_applications'),
	path('api/positions/<uuid:pk>/close', views.ClosePositionView.as_view(), name='position_close'),
	path('api/positions/<uuid:pk>/open', views.OpenPositionView.as_view(), name='position_open'),

	# UC6: Notifications
	path('api/notifications', views.notification_list_view, name='notifications_list'),
	path('api/notifications/<uuid:pk>/read', views.MarkNotificationReadView.as_view(), name='notification_mark_read'),
	path('api/notifications/read-all', views.mark_all_notifications_read, name='notifications_mark_all_read'),

	# UC7: Investor engagement
	path('api/users/favorites', views.UserFavoritesView.as_view(), name='user_favorites'),
	path('api/startups/<uuid:pk>/favorite', views.ToggleFavoriteView.as_view(), name='toggle_favorite'),
	path('api/users/interests', views.UserInterestsView.as_view(), name='user_interests'),
	path('api/startups/<uuid:pk>/interests', views.StartupInterestsView.as_view(), name='startup_interests'),
	path('api/startups/<uuid:pk>/interest', views.ExpressInterestView.as_view(), name='express_interest'),
	path('api/startups/<uuid:pk>/convert-to-marketplace', views.ConvertToMarketplaceView.as_view(), name='convert_to_marketplace'),
	
	# User management endpoints
	path('api/users/startups', views.UserStartupsView.as_view(), name='user_startups'),
	path('api/users/startups/with-applications', views.UserStartupsWithApplicationsView.as_view(), name='user_startups_with_applications'),
	
	# Statistics endpoints
	path('api/stats', views.platform_stats, name='platform_stats'),
	
	# Search endpoints
	path('api/search', views.SearchView.as_view(), name='search'),
	
	# ==================== NEW MISSING ENDPOINTS ====================
	
	# Messaging endpoints
	path('api/messages', views.ConversationListView.as_view(), name='conversations_list'),
	path('api/messages/<uuid:pk>', views.ConversationDetailView.as_view(), name='conversation_detail'),
	path('api/messages/<uuid:conversation_id>/messages', views.MessageListView.as_view(), name='messages_list'),
	path('api/messages/users/online', views.get_online_users, name='online_users'),
	
	# User profile management
	path('api/users/profile', views.UserProfileView.as_view(), name='user_profile'),
	path('api/users/profile-detail', views.UserProfileDetailView.as_view(), name='user_profile_detail'),
	path('api/users/profile-data', views.ProfileDataView.as_view(), name='user_profile_data'),
	path('account/<str:token>', views.get_account_by_token, name='account_by_token'),
	
	# File upload endpoints
	path('api/upload', views.FileUploadView.as_view(), name='file_upload'),
	path('api/upload/resume', views.upload_resume, name='upload_resume'),
	path('api/upload/startup-image', views.upload_startup_image, name='upload_startup_image'),
	path('api/upload/profile-picture', views.upload_profile_picture, name='upload_profile_picture'),
	path('api/uploads', views.FileUploadListView.as_view(), name='file_uploads_list'),
	
	# Recommendation system endpoints
	path('api/onboarding/preferences', views.OnboardingPreferencesView.as_view(), name='onboarding_preferences'),
	path('api/startups/<uuid:pk>/like', views.like_startup, name='like_startup'),
	path('api/startups/<uuid:pk>/unlike', views.unlike_startup, name='unlike_startup'),
	path('api/startups/<uuid:pk>/dislike', views.dislike_startup, name='dislike_startup'),
	path('api/startups/<uuid:pk>/undislike', views.undislike_startup, name='undislike_startup'),
	path('api/startups/<uuid:pk>/interaction-status', views.startup_interaction_status, name='startup_interaction_status'),
	
	# Trending startups endpoint
	path('api/recommendations/trending/startups', views.TrendingStartupsView.as_view(), name='trending_startups'),
	
	# Recommendation session storage
	path('api/recommendations/session', views.store_recommendation_session, name='store_recommendation_session'),
]
