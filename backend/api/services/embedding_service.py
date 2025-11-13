"""
Embedding Service for generating user and startup embeddings
"""
import json
import logging
from typing import Optional, List
from sentence_transformers import SentenceTransformer
from django.utils import timezone
from api.models import User, Startup, StartupTag
from api.messaging_models import UserProfile
from api.recommendation_models import UserOnboardingPreferences

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating embeddings using sentence-transformers"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the embedding service with a specific model
        
        Args:
            model_name: Name of the sentence-transformers model to use
        """
        self.model_name = model_name
        self._model = None
    
    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the model to avoid loading on import"""
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
        return self._model
    
    def build_profile_text(self, user: User) -> str:
        """
        Build profile text from user data for embedding generation
        
        Args:
            user: User instance
            
        Returns:
            Combined text string for embedding
        """
        profile_text_parts = []
        
        # Add role
        if user.role:
            profile_text_parts.append(f"Role: {user.role}")
        
        # Get user profile data
        try:
            profile = user.profile
            if profile.bio:
                profile_text_parts.append(f"Bio: {profile.bio}")
            if profile.skills:
                skills = profile.skills if isinstance(profile.skills, list) else json.loads(profile.skills) if isinstance(profile.skills, str) else []
                if skills:
                    profile_text_parts.append(f"Skills: {', '.join(skills)}")
            if profile.experience:
                experience = profile.experience if isinstance(profile.experience, list) else json.loads(profile.experience) if isinstance(profile.experience, str) else []
                if experience:
                    exp_text = json.dumps(experience) if isinstance(experience, list) else str(experience)
                    profile_text_parts.append(f"Experience: {exp_text}")
            if profile.location:
                profile_text_parts.append(f"Location: {profile.location}")
        except UserProfile.DoesNotExist:
            pass
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"Error parsing profile data for user {user.id}: {e}")
        
        # Get onboarding preferences
        try:
            prefs = user.onboarding_preferences
            if prefs.selected_categories:
                categories = prefs.selected_categories if isinstance(prefs.selected_categories, list) else json.loads(prefs.selected_categories) if isinstance(prefs.selected_categories, str) else []
                if categories:
                    profile_text_parts.append(f"Categories: {', '.join(categories)}")
            if prefs.selected_fields:
                fields = prefs.selected_fields if isinstance(prefs.selected_fields, list) else json.loads(prefs.selected_fields) if isinstance(prefs.selected_fields, str) else []
                if fields:
                    profile_text_parts.append(f"Fields: {', '.join(fields)}")
            if prefs.preferred_skills:
                preferred_skills = prefs.preferred_skills if isinstance(prefs.preferred_skills, list) else json.loads(prefs.preferred_skills) if isinstance(prefs.preferred_skills, str) else []
                if preferred_skills:
                    profile_text_parts.append(f"Preferred Skills: {', '.join(preferred_skills)}")
        except UserOnboardingPreferences.DoesNotExist:
            pass
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"Error parsing onboarding preferences for user {user.id}: {e}")
        
        # Join all parts with spaces
        profile_text = ' '.join(profile_text_parts)
        
        # If no text was generated, use a default
        if not profile_text.strip():
            profile_text = f"User role: {user.role or 'unknown'}"
        
        return profile_text
    
    def generate_user_embedding(self, user: User, force: bool = False) -> bool:
        """
        Generate and save embedding for a user
        
        Args:
            user: User instance
            force: If True, regenerate even if embedding exists
            
        Returns:
            True if embedding was generated successfully, False otherwise
        """
        try:
            # Build profile text
            profile_text = self.build_profile_text(user)
            
            if not profile_text.strip():
                logger.warning(f"No profile text generated for user {user.id}, skipping embedding")
                return False
            
            # Generate embedding
            logger.debug(f"Generating embedding for user {user.id}")
            embedding_vector = self.model.encode(profile_text, convert_to_numpy=True)
            
            # Validate embedding
            if embedding_vector is None or len(embedding_vector) == 0:
                logger.error(f"Empty embedding generated for user {user.id}")
                return False
            
            # Convert to list and save as JSON string
            embedding_list = embedding_vector.tolist()
            embedding_json = json.dumps(embedding_list)
            
            # Update user model
            user.profile_embedding = embedding_json
            user.embedding_model = self.model_name
            user.embedding_updated_at = timezone.now()
            user.embedding_needs_update = False
            
            # Save only the embedding-related fields
            user.save(update_fields=[
                'profile_embedding',
                'embedding_model',
                'embedding_updated_at',
                'embedding_needs_update'
            ])
            
            logger.info(f"Successfully generated embedding for user {user.id} (dimension: {len(embedding_list)})")
            return True
            
        except Exception as e:
            logger.error(f"Error generating embedding for user {user.id}: {str(e)}", exc_info=True)
            return False
    
    def generate_embeddings_batch(self, users: List[User], batch_size: int = 50) -> dict:
        """
        Generate embeddings for a batch of users
        
        Args:
            users: List of User instances
            batch_size: Number of users to process in each batch
            
        Returns:
            Dictionary with success count, failure count, and errors
        """
        results = {
            'success': 0,
            'failed': 0,
            'errors': []
        }
        
        total = len(users)
        logger.info(f"Processing {total} users in batches of {batch_size}")
        
        for i in range(0, total, batch_size):
            batch = users[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (total + batch_size - 1) // batch_size
            
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} users)")
            
            for user in batch:
                try:
                    if self.generate_user_embedding(user):
                        results['success'] += 1
                    else:
                        results['failed'] += 1
                        results['errors'].append(f"User {user.id}: Failed to generate embedding")
                except Exception as e:
                    results['failed'] += 1
                    error_msg = f"User {user.id}: {str(e)}"
                    results['errors'].append(error_msg)
                    logger.error(error_msg, exc_info=True)
        
        logger.info(f"Batch processing complete: {results['success']} succeeded, {results['failed']} failed")
        return results
    
    def build_startup_text(self, startup: Startup) -> str:
        """
        Build text from startup data for embedding generation
        
        Args:
            startup: Startup instance
            
        Returns:
            Combined text string for embedding
        """
        startup_text_parts = []
        
        # Add title
        if startup.title:
            startup_text_parts.append(f"Title: {startup.title}")
        
        # Add description
        if startup.description:
            startup_text_parts.append(f"Description: {startup.description}")
        
        # Add field
        if startup.field:
            startup_text_parts.append(f"Field: {startup.field}")
        
        # Add category
        if startup.category:
            startup_text_parts.append(f"Category: {startup.category}")
        
        # Add type
        if startup.type:
            startup_text_parts.append(f"Type: {startup.type}")
        
        # Add stages (JSON array)
        if startup.stages:
            try:
                stages = startup.stages if isinstance(startup.stages, list) else json.loads(startup.stages) if isinstance(startup.stages, str) else []
                if stages:
                    startup_text_parts.append(f"Stages: {', '.join(stages)}")
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"Error parsing stages for startup {startup.id}: {e}")
        
        # Add collaboration-specific fields
        if startup.type == 'collaboration':
            if startup.role_title:
                startup_text_parts.append(f"Role: {startup.role_title}")
            if startup.phase:
                startup_text_parts.append(f"Phase: {startup.phase}")
            if startup.team_size:
                startup_text_parts.append(f"Team Size: {startup.team_size}")
            if startup.earn_through:
                startup_text_parts.append(f"Earn Through: {startup.earn_through}")
        
        # Add tags (from StartupTag relationship)
        try:
            tags = list(StartupTag.objects.filter(startup=startup).values_list('tag', flat=True))
            if tags:
                startup_text_parts.append(f"Tags: {', '.join(tags)}")
        except Exception as e:
            logger.warning(f"Error fetching tags for startup {startup.id}: {e}")
        
        # Join all parts with spaces
        startup_text = ' '.join(startup_text_parts)
        
        # If no text was generated, use a default
        if not startup_text.strip():
            startup_text = f"Startup: {startup.title or 'unknown'} in {startup.category or 'unknown'} category"
        
        return startup_text
    
    def generate_startup_embedding(self, startup: Startup, force: bool = False) -> bool:
        """
        Generate and save embedding for a startup
        
        Args:
            startup: Startup instance
            force: If True, regenerate even if embedding exists
            
        Returns:
            True if embedding was generated successfully, False otherwise
        """
        try:
            # Build startup text
            startup_text = self.build_startup_text(startup)
            
            if not startup_text.strip():
                logger.warning(f"No startup text generated for startup {startup.id}, skipping embedding")
                return False
            
            # Generate embedding
            logger.debug(f"Generating embedding for startup {startup.id}")
            embedding_vector = self.model.encode(startup_text, convert_to_numpy=True)
            
            # Validate embedding
            if embedding_vector is None or len(embedding_vector) == 0:
                logger.error(f"Empty embedding generated for startup {startup.id}")
                return False
            
            # Convert to list and save as JSON string
            embedding_list = embedding_vector.tolist()
            embedding_json = json.dumps(embedding_list)
            
            # Update startup model
            startup.profile_embedding = embedding_json
            startup.embedding_model = self.model_name
            startup.embedding_updated_at = timezone.now()
            startup.embedding_needs_update = False
            
            # Save only the embedding-related fields
            startup.save(update_fields=[
                'profile_embedding',
                'embedding_model',
                'embedding_updated_at',
                'embedding_needs_update'
            ])
            
            logger.info(f"Successfully generated embedding for startup {startup.id} (dimension: {len(embedding_list)})")
            return True
            
        except Exception as e:
            logger.error(f"Error generating embedding for startup {startup.id}: {str(e)}", exc_info=True)
            return False
    
    def generate_startup_embeddings_batch(self, startups: List[Startup], batch_size: int = 50) -> dict:
        """
        Generate embeddings for a batch of startups
        
        Args:
            startups: List of Startup instances
            batch_size: Number of startups to process in each batch
            
        Returns:
            Dictionary with success count, failure count, and errors
        """
        results = {
            'success': 0,
            'failed': 0,
            'errors': []
        }
        
        total = len(startups)
        logger.info(f"Processing {total} startups in batches of {batch_size}")
        
        for i in range(0, total, batch_size):
            batch = startups[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (total + batch_size - 1) // batch_size
            
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} startups)")
            
            for startup in batch:
                try:
                    if self.generate_startup_embedding(startup):
                        results['success'] += 1
                    else:
                        results['failed'] += 1
                        results['errors'].append(f"Startup {startup.id}: Failed to generate embedding")
                except Exception as e:
                    results['failed'] += 1
                    error_msg = f"Startup {startup.id}: {str(e)}"
                    results['errors'].append(error_msg)
                    logger.error(error_msg, exc_info=True)
        
        logger.info(f"Batch processing complete: {results['success']} succeeded, {results['failed']} failed")
        return results

