"""
Embedding Service for generating user and startup embeddings
"""
import json
import logging
from typing import Optional, List
from sentence_transformers import SentenceTransformer
from django.utils import timezone
from api.models import User
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

