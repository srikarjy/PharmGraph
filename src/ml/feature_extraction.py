"""Modular feature engineering framework for pharmacogenomics ML pipeline."""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
import spacy
from transformers import AutoTokenizer, AutoModel
import torch

from ..config import get_logger
from ..data_ingestion.data_models import PaperMetadata
from ..quality.quality_scorer import QualityScoreComponents


logger = get_logger(__name__)


@dataclass
class FeatureMetadata:
    """Metadata for feature extraction."""
    feature_name: str
    feature_type: str  # 'text', 'numerical', 'categorical', 'embedding'
    extraction_method: str
    version: str
    created_at: datetime
    dependencies: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    quality_score: Optional[float] = None


@dataclass
class FeatureSet:
    """Container for extracted features with metadata."""
    features: Dict[str, np.ndarray]
    metadata: Dict[str, FeatureMetadata]
    feature_names: List[str]
    sample_ids: List[str]
    extraction_timestamp: datetime
    version_hash: str


class FeatureValidator:
    """Validates feature quality and consistency."""
    
    def __init__(self):
        """Initialize feature validator."""
        self.validation_rules = {
            'no_null_values': self._check_null_values,
            'numeric_range': self._check_numeric_range,
            'text_length': self._check_text_length,
            'feature_correlation': self._check_feature_correlation,
            'data_drift': self._check_data_drift
        }
    
    def validate_features(self, feature_set: FeatureSet) -> Dict[str, Any]:
        """Validate feature set quality.
        
        Args:
            feature_set: Feature set to validate
            
        Returns:
            Dict containing validation results
        """
        validation_results = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'quality_score': 100.0,
            'feature_scores': {}
        }
        
        for feature_name, feature_data in feature_set.features.items():
            feature_results = self._validate_single_feature(
                feature_name, feature_data, feature_set.metadata.get(feature_name)
            )
            
            validation_results['feature_scores'][feature_name] = feature_results
            
            if feature_results['errors']:
                validation_results['errors'].extend(feature_results['errors'])
                validation_results['is_valid'] = False
            
            if feature_results['warnings']:
                validation_results['warnings'].extend(feature_results['warnings'])
            
            # Update overall quality score
            validation_results['quality_score'] = min(
                validation_results['quality_score'],
                feature_results['quality_score']
            )
        
        return validation_results
    
    def _validate_single_feature(self, name: str, data: np.ndarray, 
                                metadata: Optional[FeatureMetadata]) -> Dict[str, Any]:
        """Validate a single feature."""
        results = {
            'quality_score': 100.0,
            'errors': [],
            'warnings': []
        }
        
        # Check for null values
        if self._check_null_values(data):
            results['errors'].append(f"Feature {name} contains null values")
            results['quality_score'] -= 20
        
        # Check numeric range for numerical features
        if metadata and metadata.feature_type == 'numerical':
            if not self._check_numeric_range(data):
                results['warnings'].append(f"Feature {name} has extreme values")
                results['quality_score'] -= 10
        
        # Check text length for text features
        if metadata and metadata.feature_type == 'text':
            if not self._check_text_length(data):
                results['warnings'].append(f"Feature {name} has inconsistent text lengths")
                results['quality_score'] -= 5
        
        return results
    
    def _check_null_values(self, data: np.ndarray) -> bool:
        """Check for null values in feature data."""
        if data.dtype == object:
            return pd.isna(data).any()
        return np.isnan(data).any()
    
    def _check_numeric_range(self, data: np.ndarray) -> bool:
        """Check if numeric values are in reasonable range."""
        if data.dtype == object:
            return True
        
        # Check for extreme outliers (beyond 3 standard deviations)
        mean_val = np.mean(data)
        std_val = np.std(data)
        outliers = np.abs(data - mean_val) > 3 * std_val
        
        return np.sum(outliers) / len(data) < 0.05  # Less than 5% outliers
    
    def _check_text_length(self, data: np.ndarray) -> bool:
        """Check text length consistency."""
        if data.dtype != object:
            return True
        
        lengths = [len(str(text)) for text in data]
        std_length = np.std(lengths)
        mean_length = np.mean(lengths)
        
        # Check if coefficient of variation is reasonable
        cv = std_length / mean_length if mean_length > 0 else 0
        return cv < 2.0  # Coefficient of variation less than 2
    
    def _check_feature_correlation(self, data: np.ndarray) -> bool:
        """Check for high correlation between features."""
        # This would be implemented for feature set correlation analysis
        return True
    
    def _check_data_drift(self, data: np.ndarray) -> bool:
        """Check for data drift in features."""
        # This would be implemented for temporal drift detection
        return True


class FeatureVersioning:
    """Manages feature versioning and lineage tracking."""
    
    def __init__(self):
        """Initialize feature versioning system."""
        self.version_history = {}
        self.lineage_graph = {}
    
    def create_version_hash(self, feature_set: FeatureSet) -> str:
        """Create version hash for feature set.
        
        Args:
            feature_set: Feature set to hash
            
        Returns:
            Version hash string
        """
        # Create hash based on feature names, metadata, and parameters
        hash_input = {
            'feature_names': sorted(feature_set.feature_names),
            'metadata': {
                name: {
                    'type': meta.feature_type,
                    'method': meta.extraction_method,
                    'version': meta.version,
                    'parameters': meta.parameters
                }
                for name, meta in feature_set.metadata.items()
            }
        }
        
        hash_string = json.dumps(hash_input, sort_keys=True)
        return hashlib.sha256(hash_string.encode()).hexdigest()[:16]
    
    def track_lineage(self, feature_name: str, dependencies: List[str], 
                     transformation: str) -> None:
        """Track feature lineage and dependencies.
        
        Args:
            feature_name: Name of the feature
            dependencies: List of dependent features/data sources
            transformation: Description of transformation applied
        """
        self.lineage_graph[feature_name] = {
            'dependencies': dependencies,
            'transformation': transformation,
            'created_at': datetime.now(),
            'version': self._get_next_version(feature_name)
        }
    
    def get_feature_lineage(self, feature_name: str) -> Dict[str, Any]:
        """Get lineage information for a feature.
        
        Args:
            feature_name: Name of the feature
            
        Returns:
            Lineage information
        """
        return self.lineage_graph.get(feature_name, {})
    
    def _get_next_version(self, feature_name: str) -> str:
        """Get next version number for feature."""
        if feature_name not in self.version_history:
            self.version_history[feature_name] = 0
        
        self.version_history[feature_name] += 1
        return f"v{self.version_history[feature_name]}"


class TextFeatureExtractor:
    """Extracts text-based features from research papers."""
    
    def __init__(self, max_features: int = 10000, use_embeddings: bool = True):
        """Initialize text feature extractor.
        
        Args:
            max_features: Maximum number of TF-IDF features
            use_embeddings: Whether to use transformer embeddings
        """
        self.max_features = max_features
        self.use_embeddings = use_embeddings
        
        # Initialize TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        
        # Initialize transformer model for embeddings
        if use_embeddings:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained('allenai/scibert-scivocab-uncased')
                self.model = AutoModel.from_pretrained('allenai/scibert-scivocab-uncased')
                self.model.eval()
            except Exception as e:
                logger.warning(f"Could not load SciBERT model: {e}")
                self.use_embeddings = False
        
        # Initialize spaCy for NER
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    def extract_tfidf_features(self, texts: List[str]) -> Tuple[np.ndarray, List[str]]:
        """Extract TF-IDF features from texts.
        
        Args:
            texts: List of text documents
            
        Returns:
            Tuple of (feature matrix, feature names)
        """
        # Fit and transform texts
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        
        return tfidf_matrix.toarray(), feature_names.tolist()
    
    def extract_embeddings(self, texts: List[str], max_length: int = 512) -> np.ndarray:
        """Extract transformer embeddings from texts.
        
        Args:
            texts: List of text documents
            max_length: Maximum sequence length
            
        Returns:
            Embedding matrix
        """
        if not self.use_embeddings:
            logger.warning("Embeddings not available, returning zeros")
            return np.zeros((len(texts), 768))
        
        embeddings = []
        
        with torch.no_grad():
            for text in texts:
                # Tokenize and encode
                inputs = self.tokenizer(
                    text, 
                    return_tensors='pt', 
                    max_length=max_length, 
                    truncation=True, 
                    padding=True
                )
                
                # Get embeddings
                outputs = self.model(**inputs)
                # Use [CLS] token embedding
                embedding = outputs.last_hidden_state[:, 0, :].numpy()
                embeddings.append(embedding.flatten())
        
        return np.array(embeddings)
    
    def extract_named_entities(self, texts: List[str]) -> Dict[str, List[int]]:
        """Extract named entities from texts.
        
        Args:
            texts: List of text documents
            
        Returns:
            Dict mapping entity types to counts per document
        """
        if not self.nlp:
            logger.warning("spaCy not available for NER")
            return {}
        
        entity_features = {
            'PERSON': [],
            'ORG': [],
            'GPE': [],  # Geopolitical entity
            'PRODUCT': [],
            'EVENT': []
        }
        
        for text in texts:
            doc = self.nlp(text)
            entity_counts = {entity_type: 0 for entity_type in entity_features.keys()}
            
            for ent in doc.ents:
                if ent.label_ in entity_counts:
                    entity_counts[ent.label_] += 1
            
            for entity_type in entity_features.keys():
                entity_features[entity_type].append(entity_counts[entity_type])
        
        return entity_features


class NumericalFeatureExtractor:
    """Extracts numerical features from paper metadata and quality scores."""
    
    def __init__(self):
        """Initialize numerical feature extractor."""
        self.scaler = StandardScaler()
        self.label_encoders = {}
    
    def extract_quality_features(self, quality_scores: List[QualityScoreComponents]) -> Dict[str, np.ndarray]:
        """Extract features from quality score components.
        
        Args:
            quality_scores: List of quality score components
            
        Returns:
            Dict of numerical features
        """
        features = {
            'journal_score': np.array([q.journal_score for q in quality_scores]),
            'clinical_relevance_score': np.array([q.clinical_relevance_score for q in quality_scores]),
            'ml_methodology_score': np.array([q.ml_methodology_score for q in quality_scores]),
            'recency_score': np.array([q.recency_score for q in quality_scores]),
            'total_score': np.array([q.total_score for q in quality_scores]),
            'confidence_level': np.array([q.confidence_level for q in quality_scores])
        }
        
        return features
    
    def extract_temporal_features(self, papers: List[PaperMetadata]) -> Dict[str, np.ndarray]:
        """Extract temporal features from publication dates.
        
        Args:
            papers: List of paper metadata
            
        Returns:
            Dict of temporal features
        """
        features = {}
        
        # Extract publication years
        pub_years = []
        pub_months = []
        days_since_pub = []
        
        current_date = datetime.now()
        
        for paper in papers:
            if paper.publication_date:
                pub_years.append(paper.publication_date.year)
                pub_months.append(paper.publication_date.month)
                
                # Days since publication
                days_diff = (current_date - paper.publication_date).days
                days_since_pub.append(days_diff)
            else:
                # Use default values for missing dates
                pub_years.append(2020)  # Default year
                pub_months.append(6)    # Default month
                days_since_pub.append(365)  # Default to 1 year ago
        
        features['publication_year'] = np.array(pub_years)
        features['publication_month'] = np.array(pub_months)
        features['days_since_publication'] = np.array(days_since_pub)
        
        # Add derived temporal features
        features['is_recent'] = (np.array(days_since_pub) < 365).astype(int)  # Published in last year
        features['publication_decade'] = (np.array(pub_years) // 10) * 10  # Decade grouping
        
        return features
    
    def extract_metadata_features(self, papers: List[PaperMetadata]) -> Dict[str, np.ndarray]:
        """Extract features from paper metadata.
        
        Args:
            papers: List of paper metadata
            
        Returns:
            Dict of metadata features
        """
        features = {}
        
        # Text length features
        title_lengths = [len(paper.title) if paper.title else 0 for paper in papers]
        abstract_lengths = [len(paper.abstract) if paper.abstract else 0 for paper in papers]
        
        features['title_length'] = np.array(title_lengths)
        features['abstract_length'] = np.array(abstract_lengths)
        features['total_text_length'] = np.array(title_lengths) + np.array(abstract_lengths)
        
        # Author count features
        author_counts = [len(paper.authors) if paper.authors else 0 for paper in papers]
        features['author_count'] = np.array(author_counts)
        
        # MeSH terms count
        mesh_counts = [len(paper.mesh_terms) if paper.mesh_terms else 0 for paper in papers]
        features['mesh_term_count'] = np.array(mesh_counts)
        
        # Keywords count
        keyword_counts = [len(paper.keywords) if paper.keywords else 0 for paper in papers]
        features['keyword_count'] = np.array(keyword_counts)
        
        # Has DOI feature
        has_doi = [1 if paper.doi else 0 for paper in papers]
        features['has_doi'] = np.array(has_doi)
        
        return features


class FeatureExtractor:
    """Main feature extraction orchestrator."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize feature extractor.
        
        Args:
            config: Configuration dictionary for feature extraction
        """
        self.config = config or {}
        
        # Initialize sub-extractors
        self.text_extractor = TextFeatureExtractor(
            max_features=self.config.get('max_tfidf_features', 10000),
            use_embeddings=self.config.get('use_embeddings', True)
        )
        self.numerical_extractor = NumericalFeatureExtractor()
        
        # Initialize validation and versioning
        self.validator = FeatureValidator()
        self.versioning = FeatureVersioning()
        
        logger.info("FeatureExtractor initialized")
    
    def extract_features(self, papers: List[PaperMetadata], 
                        quality_scores: List[QualityScoreComponents],
                        feature_types: Optional[List[str]] = None) -> FeatureSet:
        """Extract comprehensive features from papers and quality scores.
        
        Args:
            papers: List of paper metadata
            quality_scores: List of quality score components
            feature_types: Optional list of feature types to extract
            
        Returns:
            FeatureSet containing all extracted features
        """
        logger.info(f"Extracting features from {len(papers)} papers")
        
        if feature_types is None:
            feature_types = ['text', 'numerical', 'temporal', 'metadata', 'quality']
        
        features = {}
        metadata = {}
        feature_names = []
        
        # Extract text features
        if 'text' in feature_types:
            text_features, text_metadata = self._extract_text_features(papers)
            features.update(text_features)
            metadata.update(text_metadata)
            feature_names.extend(text_features.keys())
        
        # Extract numerical features
        if 'numerical' in feature_types or 'quality' in feature_types:
            numerical_features, numerical_metadata = self._extract_numerical_features(
                papers, quality_scores
            )
            features.update(numerical_features)
            metadata.update(numerical_metadata)
            feature_names.extend(numerical_features.keys())
        
        # Extract temporal features
        if 'temporal' in feature_types:
            temporal_features, temporal_metadata = self._extract_temporal_features(papers)
            features.update(temporal_features)
            metadata.update(temporal_metadata)
            feature_names.extend(temporal_features.keys())
        
        # Extract metadata features
        if 'metadata' in feature_types:
            meta_features, meta_metadata = self._extract_metadata_features(papers)
            features.update(meta_features)
            metadata.update(meta_metadata)
            feature_names.extend(meta_features.keys())
        
        # Create sample IDs
        sample_ids = [paper.pmid for paper in papers]
        
        # Create feature set
        feature_set = FeatureSet(
            features=features,
            metadata=metadata,
            feature_names=feature_names,
            sample_ids=sample_ids,
            extraction_timestamp=datetime.now(),
            version_hash=""
        )
        
        # Generate version hash
        feature_set.version_hash = self.versioning.create_version_hash(feature_set)
        
        # Validate features
        validation_results = self.validator.validate_features(feature_set)
        
        if not validation_results['is_valid']:
            logger.warning(f"Feature validation failed: {validation_results['errors']}")
        
        logger.info(f"Feature extraction completed: {len(features)} feature types, "
                   f"quality score: {validation_results['quality_score']:.1f}")
        
        return feature_set
    
    def _extract_text_features(self, papers: List[PaperMetadata]) -> Tuple[Dict[str, np.ndarray], Dict[str, FeatureMetadata]]:
        """Extract text-based features."""
        features = {}
        metadata = {}
        
        # Combine title and abstract
        texts = []
        for paper in papers:
            text = ""
            if paper.title:
                text += paper.title + " "
            if paper.abstract:
                text += paper.abstract
            texts.append(text.strip())
        
        # Extract TF-IDF features
        tfidf_matrix, tfidf_names = self.text_extractor.extract_tfidf_features(texts)
        
        # Store TF-IDF features (flatten to 1D per sample)
        for i, name in enumerate(tfidf_names[:100]):  # Limit to top 100 features
            feature_name = f"tfidf_{name}"
            features[feature_name] = tfidf_matrix[:, i]
            metadata[feature_name] = FeatureMetadata(
                feature_name=feature_name,
                feature_type='numerical',
                extraction_method='tfidf',
                version='v1',
                created_at=datetime.now(),
                parameters={'max_features': self.text_extractor.max_features}
            )
        
        # Extract embeddings if available
        if self.text_extractor.use_embeddings:
            embeddings = self.text_extractor.extract_embeddings(texts)
            
            # Store embedding dimensions
            for i in range(min(50, embeddings.shape[1])):  # Limit to 50 dimensions
                feature_name = f"embedding_dim_{i}"
                features[feature_name] = embeddings[:, i]
                metadata[feature_name] = FeatureMetadata(
                    feature_name=feature_name,
                    feature_type='numerical',
                    extraction_method='scibert_embedding',
                    version='v1',
                    created_at=datetime.now(),
                    parameters={'model': 'allenai/scibert-scivocab-uncased'}
                )
        
        # Extract named entities
        entity_features = self.text_extractor.extract_named_entities(texts)
        for entity_type, counts in entity_features.items():
            feature_name = f"entity_{entity_type.lower()}_count"
            features[feature_name] = np.array(counts)
            metadata[feature_name] = FeatureMetadata(
                feature_name=feature_name,
                feature_type='numerical',
                extraction_method='spacy_ner',
                version='v1',
                created_at=datetime.now()
            )
        
        return features, metadata
    
    def _extract_numerical_features(self, papers: List[PaperMetadata], 
                                  quality_scores: List[QualityScoreComponents]) -> Tuple[Dict[str, np.ndarray], Dict[str, FeatureMetadata]]:
        """Extract numerical features."""
        features = {}
        metadata = {}
        
        # Extract quality features
        quality_features = self.numerical_extractor.extract_quality_features(quality_scores)
        
        for name, values in quality_features.items():
            features[name] = values
            metadata[name] = FeatureMetadata(
                feature_name=name,
                feature_type='numerical',
                extraction_method='quality_scorer',
                version='v1',
                created_at=datetime.now()
            )
        
        return features, metadata
    
    def _extract_temporal_features(self, papers: List[PaperMetadata]) -> Tuple[Dict[str, np.ndarray], Dict[str, FeatureMetadata]]:
        """Extract temporal features."""
        features = {}
        metadata = {}
        
        temporal_features = self.numerical_extractor.extract_temporal_features(papers)
        
        for name, values in temporal_features.items():
            features[name] = values
            metadata[name] = FeatureMetadata(
                feature_name=name,
                feature_type='numerical',
                extraction_method='temporal_extraction',
                version='v1',
                created_at=datetime.now()
            )
        
        return features, metadata
    
    def _extract_metadata_features(self, papers: List[PaperMetadata]) -> Tuple[Dict[str, np.ndarray], Dict[str, FeatureMetadata]]:
        """Extract metadata features."""
        features = {}
        metadata = {}
        
        meta_features = self.numerical_extractor.extract_metadata_features(papers)
        
        for name, values in meta_features.items():
            features[name] = values
            metadata[name] = FeatureMetadata(
                feature_name=name,
                feature_type='numerical',
                extraction_method='metadata_extraction',
                version='v1',
                created_at=datetime.now()
            )
        
        return features, metadata
    
    def save_features(self, feature_set: FeatureSet, filepath: str) -> None:
        """Save feature set to file.
        
        Args:
            feature_set: Feature set to save
            filepath: Path to save file
        """
        # Convert to pandas DataFrame for easy saving
        df_data = {}
        
        # Add sample IDs
        df_data['sample_id'] = feature_set.sample_ids
        
        # Add all features
        for name, values in feature_set.features.items():
            df_data[name] = values
        
        df = pd.DataFrame(df_data)
        
        # Save as parquet for efficiency
        df.to_parquet(filepath, index=False)
        
        # Save metadata separately
        metadata_path = filepath.replace('.parquet', '_metadata.json')
        metadata_dict = {
            name: {
                'feature_name': meta.feature_name,
                'feature_type': meta.feature_type,
                'extraction_method': meta.extraction_method,
                'version': meta.version,
                'created_at': meta.created_at.isoformat(),
                'parameters': meta.parameters
            }
            for name, meta in feature_set.metadata.items()
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata_dict, f, indent=2)
        
        logger.info(f"Features saved to {filepath}")
    
    def load_features(self, filepath: str) -> FeatureSet:
        """Load feature set from file.
        
        Args:
            filepath: Path to feature file
            
        Returns:
            Loaded feature set
        """
        # Load DataFrame
        df = pd.read_parquet(filepath)
        
        # Extract sample IDs
        sample_ids = df['sample_id'].tolist()
        
        # Extract features
        feature_columns = [col for col in df.columns if col != 'sample_id']
        features = {col: df[col].values for col in feature_columns}
        
        # Load metadata
        metadata_path = filepath.replace('.parquet', '_metadata.json')
        metadata = {}
        
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata_dict = json.load(f)
            
            for name, meta_dict in metadata_dict.items():
                metadata[name] = FeatureMetadata(
                    feature_name=meta_dict['feature_name'],
                    feature_type=meta_dict['feature_type'],
                    extraction_method=meta_dict['extraction_method'],
                    version=meta_dict['version'],
                    created_at=datetime.fromisoformat(meta_dict['created_at']),
                    parameters=meta_dict.get('parameters', {})
                )
        
        feature_set = FeatureSet(
            features=features,
            metadata=metadata,
            feature_names=feature_columns,
            sample_ids=sample_ids,
            extraction_timestamp=datetime.now(),
            version_hash=""
        )
        
        # Generate version hash
        feature_set.version_hash = self.versioning.create_version_hash(feature_set)
        
        logger.info(f"Features loaded from {filepath}")
        return feature_set