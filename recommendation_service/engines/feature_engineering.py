"""
Feature Engineering for Two-Tower Model
Handles embedding processing, categorical encoding, and feature preparation
"""
import numpy as np
import json
import pickle
from typing import Dict, List, Tuple, Any
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from utils.logger import get_logger

logger = get_logger(__name__)


class FeatureEncoder:
    """Encodes categorical features for model training"""
    
    def __init__(self):
        self.role_encoder = {}
        self.type_encoder = {}
        self.category_encoder = {}
        self.field_encoder = {}
        self.phase_encoder = {}
        self.categories_mlb = MultiLabelBinarizer()
        self.fields_mlb = MultiLabelBinarizer()
        self.tags_mlb = MultiLabelBinarizer()
        self.stages_mlb = MultiLabelBinarizer()
        self.engagement_mlb = MultiLabelBinarizer()
        self.skills_mlb = MultiLabelBinarizer()
        self.startup_tags_mlb = MultiLabelBinarizer()
        self.startup_stages_mlb = MultiLabelBinarizer()
        self.is_fitted = False
    
    def fit(self, data: List[Dict]):
        """Fit encoders on training data"""
        logger.info("Fitting feature encoders...")
        
        # Extract unique values for single-value categorical features
        roles = list(set([d['user_role'] for d in data]))
        types = list(set([d['startup_type'] for d in data]))
        categories = list(set([d['startup_category'] for d in data]))
        fields = list(set([d['startup_field'] for d in data]))
        phases = list(set([d.get('startup_phase', '') for d in data if d.get('startup_phase')]))
        
        # Create label encoders (simple dict mapping)
        self.role_encoder = {role: idx for idx, role in enumerate(sorted(roles))}
        self.type_encoder = {t: idx for idx, t in enumerate(sorted(types))}
        self.category_encoder = {cat: idx for idx, cat in enumerate(sorted(categories))}
        self.field_encoder = {field: idx for idx, field in enumerate(sorted(fields))}
        self.phase_encoder = {phase: idx for idx, phase in enumerate(sorted(phases))}
        
        # Fit multi-label binarizers
        user_categories = [self._parse_json_list(d.get('user_categories', '[]')) for d in data]
        user_fields = [self._parse_json_list(d.get('user_fields', '[]')) for d in data]
        user_tags = [self._parse_json_list(d.get('user_tags', '[]')) for d in data]
        user_stages = [self._parse_json_list(d.get('user_stages', '[]')) for d in data]
        user_engagement = [self._parse_json_list(d.get('user_engagement', '[]')) for d in data]
        user_skills = [self._parse_json_list(d.get('user_skills', '[]')) for d in data]
        startup_tags = [self._parse_json_list(d.get('startup_tags', '[]')) for d in data]
        startup_stages = [self._parse_json_list(d.get('startup_stages', '[]')) for d in data]
        
        self.categories_mlb.fit(user_categories)
        self.fields_mlb.fit(user_fields)
        self.tags_mlb.fit(user_tags)
        self.stages_mlb.fit(user_stages)
        self.engagement_mlb.fit(user_engagement)
        self.skills_mlb.fit(user_skills)
        self.startup_tags_mlb.fit(startup_tags)
        self.startup_stages_mlb.fit(startup_stages)
        
        self.is_fitted = True
        
        logger.info(f"Encoders fitted:")
        logger.info(f"  Roles: {len(self.role_encoder)}")
        logger.info(f"  Types: {len(self.type_encoder)}")
        logger.info(f"  Categories: {len(self.category_encoder)}")
        logger.info(f"  Fields: {len(self.field_encoder)}")
        logger.info(f"  User Categories Dim: {len(self.categories_mlb.classes_)}")
        logger.info(f"  User Tags Dim: {len(self.tags_mlb.classes_)}")
        logger.info(f"  User Skills Dim: {len(self.skills_mlb.classes_)}")
        logger.info(f"  Startup Tags Dim: {len(self.startup_tags_mlb.classes_)}")
    
    def transform_user_features(self, sample: Dict) -> Dict[str, np.ndarray]:
        """Transform user features for a single sample"""
        if not self.is_fitted:
            raise ValueError("Encoder must be fitted before transform")
        
        features = {}
        
        # Embedding
        embedding = self._parse_embedding(sample.get('user_embedding'))
        if embedding is not None:
            features['user_embedding'] = np.array(embedding, dtype=np.float32)
        else:
            features['user_embedding'] = np.zeros(384, dtype=np.float32)
        
        # Role (one-hot)
        role = sample.get('user_role', '')
        role_idx = self.role_encoder.get(role, 0)
        role_onehot = np.zeros(len(self.role_encoder), dtype=np.float32)
        role_onehot[role_idx] = 1.0
        features['user_role'] = role_onehot
        
        # Multi-label features
        features['user_categories'] = self.categories_mlb.transform(
            [self._parse_json_list(sample.get('user_categories', '[]'))]
        )[0].astype(np.float32)
        
        features['user_fields'] = self.fields_mlb.transform(
            [self._parse_json_list(sample.get('user_fields', '[]'))]
        )[0].astype(np.float32)
        
        features['user_tags'] = self.tags_mlb.transform(
            [self._parse_json_list(sample.get('user_tags', '[]'))]
        )[0].astype(np.float32)
        
        features['user_stages'] = self.stages_mlb.transform(
            [self._parse_json_list(sample.get('user_stages', '[]'))]
        )[0].astype(np.float32)
        
        features['user_engagement'] = self.engagement_mlb.transform(
            [self._parse_json_list(sample.get('user_engagement', '[]'))]
        )[0].astype(np.float32)
        
        features['user_skills'] = self.skills_mlb.transform(
            [self._parse_json_list(sample.get('user_skills', '[]'))]
        )[0].astype(np.float32)
        
        return features
    
    def transform_startup_features(self, sample: Dict) -> Dict[str, np.ndarray]:
        """Transform startup features for a single sample"""
        if not self.is_fitted:
            raise ValueError("Encoder must be fitted before transform")
        
        features = {}
        
        # Embedding
        embedding = self._parse_embedding(sample.get('startup_embedding'))
        if embedding is not None:
            features['startup_embedding'] = np.array(embedding, dtype=np.float32)
        else:
            features['startup_embedding'] = np.zeros(384, dtype=np.float32)
        
        # Type (one-hot)
        stype = sample.get('startup_type', '')
        type_idx = self.type_encoder.get(stype, 0)
        type_onehot = np.zeros(len(self.type_encoder), dtype=np.float32)
        type_onehot[type_idx] = 1.0
        features['startup_type'] = type_onehot
        
        # Category (one-hot)
        category = sample.get('startup_category', '')
        cat_idx = self.category_encoder.get(category, 0)
        cat_onehot = np.zeros(len(self.category_encoder), dtype=np.float32)
        cat_onehot[cat_idx] = 1.0
        features['startup_category'] = cat_onehot
        
        # Field (one-hot)
        field = sample.get('startup_field', '')
        field_idx = self.field_encoder.get(field, 0)
        field_onehot = np.zeros(len(self.field_encoder), dtype=np.float32)
        field_onehot[field_idx] = 1.0
        features['startup_field'] = field_onehot
        
        # Phase (one-hot)
        phase = sample.get('startup_phase', '')
        if phase and phase in self.phase_encoder:
            phase_idx = self.phase_encoder[phase]
            phase_onehot = np.zeros(len(self.phase_encoder), dtype=np.float32)
            phase_onehot[phase_idx] = 1.0
        else:
            phase_onehot = np.zeros(len(self.phase_encoder) if self.phase_encoder else 1, dtype=np.float32)
        features['startup_phase'] = phase_onehot
        
        # Multi-label features
        features['startup_tags'] = self.startup_tags_mlb.transform(
            [self._parse_json_list(sample.get('startup_tags', '[]'))]
        )[0].astype(np.float32)
        
        features['startup_stages'] = self.startup_stages_mlb.transform(
            [self._parse_json_list(sample.get('startup_stages', '[]'))]
        )[0].astype(np.float32)
        
        return features
    
    def get_user_feature_dim(self) -> int:
        """Get total dimension of user features"""
        if not self.is_fitted:
            return 0
        
        dim = 384  # embedding
        dim += len(self.role_encoder)  # role one-hot
        dim += len(self.categories_mlb.classes_)
        dim += len(self.fields_mlb.classes_)
        dim += len(self.tags_mlb.classes_)
        dim += len(self.stages_mlb.classes_)
        dim += len(self.engagement_mlb.classes_)
        dim += len(self.skills_mlb.classes_)
        return dim
    
    def get_startup_feature_dim(self) -> int:
        """Get total dimension of startup features"""
        if not self.is_fitted:
            return 0
        
        dim = 384  # embedding
        dim += len(self.type_encoder)  # type one-hot
        dim += len(self.category_encoder)  # category one-hot
        dim += len(self.field_encoder)  # field one-hot
        dim += len(self.phase_encoder) if self.phase_encoder else 1  # phase one-hot
        dim += len(self.startup_tags_mlb.classes_)
        dim += len(self.startup_stages_mlb.classes_)
        return dim
    
    def save(self, filepath: str):
        """Save encoder to file"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        logger.info(f"Encoder saved to {filepath}")
    
    @staticmethod
    def load(filepath: str) -> 'FeatureEncoder':
        """Load encoder from file"""
        with open(filepath, 'rb') as f:
            encoder = pickle.load(f)
        logger.info(f"Encoder loaded from {filepath}")
        return encoder
    
    def _parse_json_list(self, json_str: str) -> List:
        """Parse JSON list string"""
        if not json_str or json_str == 'null':
            return []
        try:
            data = json.loads(json_str)
            return data if isinstance(data, list) else []
        except:
            return []
    
    def _parse_embedding(self, emb_str: str) -> List[float]:
        """Parse embedding JSON string"""
        if not emb_str or emb_str == 'null':
            return None
        try:
            emb = json.loads(emb_str)
            if isinstance(emb, list) and len(emb) > 0:
                return emb
        except:
            pass
        return None


class DatasetProcessor:
    """Process raw dataset into training/validation/test splits"""
    
    def __init__(self, encoder: FeatureEncoder = None):
        self.encoder = encoder if encoder else FeatureEncoder()
    
    def prepare_dataset(
        self, 
        data: List[Dict],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42
    ) -> Tuple[Dict, Dict, Dict]:
        """
        Prepare dataset with train/val/test splits
        
        Args:
            data: List of sample dictionaries
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            random_seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_data, val_data, test_data) dictionaries
        """
        logger.info(f"Preparing dataset with {len(data)} samples...")
        
        # Fit encoder on training data
        if not self.encoder.is_fitted:
            self.encoder.fit(data)
        
        # Split data
        train_data, temp_data = train_test_split(
            data, 
            test_size=(val_ratio + test_ratio),
            random_state=random_seed,
            shuffle=True
        )
        
        val_size = val_ratio / (val_ratio + test_ratio)
        val_data, test_data = train_test_split(
            temp_data,
            test_size=(1 - val_size),
            random_state=random_seed,
            shuffle=True
        )
        
        logger.info(f"Split: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")
        
        # Process each split
        train_processed = self._process_split(train_data, 'train')
        val_processed = self._process_split(val_data, 'val')
        test_processed = self._process_split(test_data, 'test')
        
        return train_processed, val_processed, test_processed
    
    def _process_split(self, data: List[Dict], split_name: str) -> Dict:
        """Process a data split"""
        logger.info(f"Processing {split_name} split...")
        
        user_features_list = []
        startup_features_list = []
        labels = []
        weights = []
        
        for sample in data:
            # Extract features
            user_feats = self.encoder.transform_user_features(sample)
            startup_feats = self.encoder.transform_startup_features(sample)
            
            # Concatenate user features
            user_feat_vector = np.concatenate([
                user_feats['user_embedding'],
                user_feats['user_role'],
                user_feats['user_categories'],
                user_feats['user_fields'],
                user_feats['user_tags'],
                user_feats['user_stages'],
                user_feats['user_engagement'],
                user_feats['user_skills'],
            ])
            
            # Concatenate startup features
            startup_feat_vector = np.concatenate([
                startup_feats['startup_embedding'],
                startup_feats['startup_type'],
                startup_feats['startup_category'],
                startup_feats['startup_field'],
                startup_feats['startup_phase'],
                startup_feats['startup_tags'],
                startup_feats['startup_stages'],
            ])
            
            user_features_list.append(user_feat_vector)
            startup_features_list.append(startup_feat_vector)
            labels.append(sample['label'])
            weights.append(sample['weight'])
        
        # Convert to numpy arrays
        user_features = np.array(user_features_list, dtype=np.float32)
        startup_features = np.array(startup_features_list, dtype=np.float32)
        labels = np.array(labels, dtype=np.float32)
        weights = np.array(weights, dtype=np.float32)
        
        logger.info(f"  User features shape: {user_features.shape}")
        logger.info(f"  Startup features shape: {startup_features.shape}")
        logger.info(f"  Labels shape: {labels.shape}")
        
        return {
            'user_features': user_features,
            'startup_features': startup_features,
            'labels': labels,
            'weights': weights,
        }


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """L2 normalize embeddings"""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
    return embeddings / norms


def load_dataset_from_csv(filepath: str) -> List[Dict]:
    """Load dataset from CSV file"""
    import csv
    
    logger.info(f"Loading dataset from {filepath}...")
    
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert label and weight to float
            row['label'] = float(row['label'])
            row['weight'] = float(row['weight'])
            data.append(row)
    
    logger.info(f"Loaded {len(data)} samples")
    return data

