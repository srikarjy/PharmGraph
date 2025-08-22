"""Advanced NLP features for pharmacogenomics text analysis."""

import re
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
import spacy
from spacy import displacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel, pipeline
import torch
from collections import Counter, defaultdict
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer

from ..config import get_logger
from ..data_ingestion.data_models import PaperMetadata


logger = get_logger(__name__)


@dataclass
class BiomedicalEntity:
    """Biomedical named entity with metadata."""
    text: str
    label: str
    start: int
    end: int
    confidence: float
    entity_id: Optional[str] = None
    synonyms: List[str] = None


@dataclass
class TextFeatures:
    """Container for advanced text features."""
    tfidf_features: np.ndarray
    embeddings: np.ndarray
    biomedical_entities: List[BiomedicalEntity]
    topic_distributions: np.ndarray
    semantic_similarity: np.ndarray
    linguistic_features: Dict[str, np.ndarray]
    citation_features: Dict[str, np.ndarray]
    equation_features: Dict[str, np.ndarray]


class ScientificTextPreprocessor:
    """Preprocesses scientific text for feature extraction."""
    
    def __init__(self):
        """Initialize scientific text preprocessor."""
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')
        
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Scientific text patterns
        self.citation_pattern = re.compile(r'\[(\d+(?:,\s*\d+)*)\]|\(([^)]+,\s*\d{4})\)')
        self.equation_pattern = re.compile(r'[A-Za-z]\s*=\s*[^.]+|∑|∫|∂|α|β|γ|δ|ε|θ|λ|μ|π|σ|φ|χ|ψ|ω')
        self.reference_pattern = re.compile(r'et\s+al\.?|doi:|DOI:|PMID:|pmid:')
        self.statistical_pattern = re.compile(r'p\s*[<>=]\s*0\.\d+|r\s*=\s*0\.\d+|n\s*=\s*\d+')
        
        logger.info("ScientificTextPreprocessor initialized")
    
    def preprocess_text(self, text: str) -> Dict[str, Any]:
        """Preprocess scientific text and extract structural features.
        
        Args:
            text: Input scientific text
            
        Returns:
            Dict containing preprocessed text and extracted features
        """
        if not text:
            return {
                'cleaned_text': '',
                'citations': [],
                'equations': [],
                'references': [],
                'statistical_mentions': [],
                'sentence_count': 0,
                'avg_sentence_length': 0
            }
        
        # Extract citations
        citations = self._extract_citations(text)
        
        # Extract equations
        equations = self._extract_equations(text)
        
        # Extract references
        references = self._extract_references(text)
        
        # Extract statistical mentions
        statistical_mentions = self._extract_statistical_mentions(text)
        
        # Clean text
        cleaned_text = self._clean_scientific_text(text)
        
        # Calculate sentence-level features
        sentences = sent_tokenize(cleaned_text)
        sentence_count = len(sentences)
        avg_sentence_length = np.mean([len(word_tokenize(sent)) for sent in sentences]) if sentences else 0
        
        return {
            'cleaned_text': cleaned_text,
            'citations': citations,
            'equations': equations,
            'references': references,
            'statistical_mentions': statistical_mentions,
            'sentence_count': sentence_count,
            'avg_sentence_length': avg_sentence_length
        }
    
    def _extract_citations(self, text: str) -> List[str]:
        """Extract citation patterns from text."""
        citations = []
        for match in self.citation_pattern.finditer(text):
            citations.append(match.group())
        return citations
    
    def _extract_equations(self, text: str) -> List[str]:
        """Extract equation patterns from text."""
        equations = []
        for match in self.equation_pattern.finditer(text):
            equations.append(match.group())
        return equations
    
    def _extract_references(self, text: str) -> List[str]:
        """Extract reference patterns from text."""
        references = []
        for match in self.reference_pattern.finditer(text):
            references.append(match.group())
        return references
    
    def _extract_statistical_mentions(self, text: str) -> List[str]:
        """Extract statistical mentions from text."""
        statistical = []
        for match in self.statistical_pattern.finditer(text):
            statistical.append(match.group())
        return statistical
    
    def _clean_scientific_text(self, text: str) -> str:
        """Clean scientific text for NLP processing."""
        # Remove citations
        text = self.citation_pattern.sub(' ', text)
        
        # Remove equations (keep for separate analysis)
        text = self.equation_pattern.sub(' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep scientific notation
        text = re.sub(r'[^\w\s\-\.\,\;\:\(\)]', ' ', text)
        
        return text.strip()


class BiomedicalNER:
    """Biomedical Named Entity Recognition using specialized models."""
    
    def __init__(self):
        """Initialize biomedical NER."""
        # Initialize spaCy with biomedical model if available
        try:
            self.nlp = spacy.load("en_core_sci_sm")  # ScispaCy model
            logger.info("Loaded ScispaCy biomedical model")
        except OSError:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                logger.warning("ScispaCy not available, using standard spaCy model")
            except OSError:
                logger.error("No spaCy model available")
                self.nlp = None
        
        # Initialize biomedical NER pipeline
        try:
            self.bio_ner = pipeline(
                "ner",
                model="d4data/biomedical-ner-all",
                tokenizer="d4data/biomedical-ner-all",
                aggregation_strategy="simple"
            )
            logger.info("Loaded biomedical NER pipeline")
        except Exception as e:
            logger.warning(f"Could not load biomedical NER pipeline: {e}")
            self.bio_ner = None
        
        # Pharmacogenomics-specific entity patterns
        self.drug_patterns = [
            r'\b(warfarin|clopidogrel|codeine|tramadol|omeprazole|simvastatin|atorvastatin)\b',
            r'\b\w+mycin\b',  # Antibiotics ending in -mycin
            r'\b\w+prazole\b',  # Proton pump inhibitors
            r'\b\w+statin\b',  # Statins
        ]
        
        self.gene_patterns = [
            r'\bCYP[0-9][A-Z][0-9]+\b',  # CYP enzymes
            r'\b[A-Z]{2,6}[0-9]*\b',  # Gene symbols
            r'\brs[0-9]+\b',  # SNP identifiers
        ]
        
        self.disease_patterns = [
            r'\b(hypertension|diabetes|cancer|depression|schizophrenia|epilepsy)\b',
            r'\b\w+oma\b',  # Tumors ending in -oma
            r'\b\w+itis\b',  # Inflammatory conditions
        ]
    
    def extract_biomedical_entities(self, text: str) -> List[BiomedicalEntity]:
        """Extract biomedical entities from text.
        
        Args:
            text: Input text
            
        Returns:
            List of biomedical entities
        """
        entities = []
        
        # Use biomedical NER pipeline if available
        if self.bio_ner:
            try:
                bio_entities = self.bio_ner(text)
                for entity in bio_entities:
                    entities.append(BiomedicalEntity(
                        text=entity['word'],
                        label=entity['entity_group'],
                        start=entity['start'],
                        end=entity['end'],
                        confidence=entity['score']
                    ))
            except Exception as e:
                logger.warning(f"Biomedical NER failed: {e}")
        
        # Use spaCy NER as fallback
        if self.nlp:
            doc = self.nlp(text)
            for ent in doc.ents:
                entities.append(BiomedicalEntity(
                    text=ent.text,
                    label=ent.label_,
                    start=ent.start_char,
                    end=ent.end_char,
                    confidence=1.0  # spaCy doesn't provide confidence scores
                ))
        
        # Add pattern-based entity extraction
        pattern_entities = self._extract_pattern_entities(text)
        entities.extend(pattern_entities)
        
        return entities
    
    def _extract_pattern_entities(self, text: str) -> List[BiomedicalEntity]:
        """Extract entities using regex patterns."""
        entities = []
        
        # Extract drugs
        for pattern in self.drug_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append(BiomedicalEntity(
                    text=match.group(),
                    label='DRUG',
                    start=match.start(),
                    end=match.end(),
                    confidence=0.8
                ))
        
        # Extract genes
        for pattern in self.gene_patterns:
            for match in re.finditer(pattern, text):
                entities.append(BiomedicalEntity(
                    text=match.group(),
                    label='GENE',
                    start=match.start(),
                    end=match.end(),
                    confidence=0.8
                ))
        
        # Extract diseases
        for pattern in self.disease_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append(BiomedicalEntity(
                    text=match.group(),
                    label='DISEASE',
                    start=match.start(),
                    end=match.end(),
                    confidence=0.8
                ))
        
        return entities


class SemanticSimilarityAnalyzer:
    """Analyzes semantic similarity using pre-trained models."""
    
    def __init__(self, model_name: str = "allenai/scibert-scivocab-uncased"):
        """Initialize semantic similarity analyzer.
        
        Args:
            model_name: Name of the pre-trained model to use
        """
        self.model_name = model_name
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.eval()
            logger.info(f"Loaded model: {model_name}")
        except Exception as e:
            logger.error(f"Could not load model {model_name}: {e}")
            self.tokenizer = None
            self.model = None
    
    def compute_embeddings(self, texts: List[str], max_length: int = 512) -> np.ndarray:
        """Compute embeddings for texts.
        
        Args:
            texts: List of input texts
            max_length: Maximum sequence length
            
        Returns:
            Embedding matrix
        """
        if not self.model:
            logger.warning("Model not available, returning zeros")
            return np.zeros((len(texts), 768))
        
        embeddings = []
        
        with torch.no_grad():
            for text in texts:
                # Tokenize
                inputs = self.tokenizer(
                    text,
                    return_tensors='pt',
                    max_length=max_length,
                    truncation=True,
                    padding=True
                )
                
                # Get embeddings
                outputs = self.model(**inputs)
                # Use mean pooling of all tokens
                embedding = outputs.last_hidden_state.mean(dim=1).numpy()
                embeddings.append(embedding.flatten())
        
        return np.array(embeddings)
    
    def compute_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute pairwise similarity matrix.
        
        Args:
            embeddings: Embedding matrix
            
        Returns:
            Similarity matrix
        """
        return cosine_similarity(embeddings)
    
    def find_similar_papers(self, query_embedding: np.ndarray, 
                           paper_embeddings: np.ndarray, 
                           top_k: int = 10) -> List[Tuple[int, float]]:
        """Find most similar papers to query.
        
        Args:
            query_embedding: Query paper embedding
            paper_embeddings: All paper embeddings
            top_k: Number of similar papers to return
            
        Returns:
            List of (index, similarity_score) tuples
        """
        similarities = cosine_similarity([query_embedding], paper_embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        return [(idx, similarities[idx]) for idx in top_indices]


class TopicModelingAnalyzer:
    """Performs topic modeling for research categorization."""
    
    def __init__(self, n_topics: int = 20, random_state: int = 42):
        """Initialize topic modeling analyzer.
        
        Args:
            n_topics: Number of topics to extract
            random_state: Random state for reproducibility
        """
        self.n_topics = n_topics
        self.random_state = random_state
        
        # Initialize vectorizer for topic modeling
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        
        # Initialize LDA model
        self.lda_model = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=random_state,
            max_iter=10,
            learning_method='online'
        )
        
        self.is_fitted = False
    
    def fit_topic_model(self, texts: List[str]) -> None:
        """Fit topic model on texts.
        
        Args:
            texts: List of input texts
        """
        # Vectorize texts
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        
        # Fit LDA model
        self.lda_model.fit(tfidf_matrix)
        self.is_fitted = True
        
        logger.info(f"Topic model fitted with {self.n_topics} topics")
    
    def get_topic_distributions(self, texts: List[str]) -> np.ndarray:
        """Get topic distributions for texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            Topic distribution matrix
        """
        if not self.is_fitted:
            logger.warning("Topic model not fitted, fitting on provided texts")
            self.fit_topic_model(texts)
        
        # Transform texts
        tfidf_matrix = self.vectorizer.transform(texts)
        topic_distributions = self.lda_model.transform(tfidf_matrix)
        
        return topic_distributions
    
    def get_top_words_per_topic(self, n_words: int = 10) -> Dict[int, List[str]]:
        """Get top words for each topic.
        
        Args:
            n_words: Number of top words per topic
            
        Returns:
            Dict mapping topic index to top words
        """
        if not self.is_fitted:
            logger.error("Topic model not fitted")
            return {}
        
        feature_names = self.vectorizer.get_feature_names_out()
        topic_words = {}
        
        for topic_idx, topic in enumerate(self.lda_model.components_):
            top_word_indices = topic.argsort()[::-1][:n_words]
            top_words = [feature_names[i] for i in top_word_indices]
            topic_words[topic_idx] = top_words
        
        return topic_words


class AdvancedTextFeatureExtractor:
    """Main class for advanced text feature extraction."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize advanced text feature extractor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Initialize components
        self.preprocessor = ScientificTextPreprocessor()
        self.ner = BiomedicalNER()
        self.similarity_analyzer = SemanticSimilarityAnalyzer(
            model_name=self.config.get('embedding_model', 'allenai/scibert-scivocab-uncased')
        )
        self.topic_analyzer = TopicModelingAnalyzer(
            n_topics=self.config.get('n_topics', 20)
        )
        
        logger.info("AdvancedTextFeatureExtractor initialized")
    
    def extract_features(self, papers: List[PaperMetadata]) -> TextFeatures:
        """Extract comprehensive text features from papers.
        
        Args:
            papers: List of paper metadata
            
        Returns:
            TextFeatures object containing all extracted features
        """
        logger.info(f"Extracting advanced text features from {len(papers)} papers")
        
        # Prepare texts
        texts = []
        all_entities = []
        linguistic_features = defaultdict(list)
        citation_features = defaultdict(list)
        equation_features = defaultdict(list)
        
        for paper in papers:
            # Combine title and abstract
            text = ""
            if paper.title:
                text += paper.title + " "
            if paper.abstract:
                text += paper.abstract
            
            texts.append(text.strip())
            
            # Preprocess text
            preprocessed = self.preprocessor.preprocess_text(text)
            
            # Extract biomedical entities
            entities = self.ner.extract_biomedical_entities(text)
            all_entities.append(entities)
            
            # Collect linguistic features
            linguistic_features['sentence_count'].append(preprocessed['sentence_count'])
            linguistic_features['avg_sentence_length'].append(preprocessed['avg_sentence_length'])
            
            # Collect citation features
            citation_features['citation_count'].append(len(preprocessed['citations']))
            citation_features['reference_count'].append(len(preprocessed['references']))
            citation_features['statistical_mention_count'].append(len(preprocessed['statistical_mentions']))
            
            # Collect equation features
            equation_features['equation_count'].append(len(preprocessed['equations']))
        
        # Extract TF-IDF features
        vectorizer = TfidfVectorizer(
            max_features=self.config.get('max_tfidf_features', 1000),
            stop_words='english',
            ngram_range=(1, 2)
        )
        tfidf_features = vectorizer.fit_transform(texts).toarray()
        
        # Extract embeddings
        embeddings = self.similarity_analyzer.compute_embeddings(texts)
        
        # Extract topic distributions
        topic_distributions = self.topic_analyzer.get_topic_distributions(texts)
        
        # Compute semantic similarity matrix
        semantic_similarity = self.similarity_analyzer.compute_similarity_matrix(embeddings)
        
        # Convert feature collections to numpy arrays
        linguistic_arrays = {k: np.array(v) for k, v in linguistic_features.items()}
        citation_arrays = {k: np.array(v) for k, v in citation_features.items()}
        equation_arrays = {k: np.array(v) for k, v in equation_features.items()}
        
        # Create TextFeatures object
        text_features = TextFeatures(
            tfidf_features=tfidf_features,
            embeddings=embeddings,
            biomedical_entities=all_entities,
            topic_distributions=topic_distributions,
            semantic_similarity=semantic_similarity,
            linguistic_features=linguistic_arrays,
            citation_features=citation_arrays,
            equation_features=equation_arrays
        )
        
        logger.info("Advanced text feature extraction completed")
        return text_features
    
    def extract_entity_features(self, papers: List[PaperMetadata]) -> Dict[str, np.ndarray]:
        """Extract entity-based features.
        
        Args:
            papers: List of paper metadata
            
        Returns:
            Dict of entity feature arrays
        """
        entity_features = defaultdict(list)
        
        for paper in papers:
            text = ""
            if paper.title:
                text += paper.title + " "
            if paper.abstract:
                text += paper.abstract
            
            entities = self.ner.extract_biomedical_entities(text)
            
            # Count entities by type
            entity_counts = Counter([entity.label for entity in entities])
            
            # Standard entity types
            for entity_type in ['DRUG', 'GENE', 'DISEASE', 'PERSON', 'ORG']:
                entity_features[f'{entity_type.lower()}_count'].append(
                    entity_counts.get(entity_type, 0)
                )
            
            # Total entity count
            entity_features['total_entity_count'].append(len(entities))
            
            # Average entity confidence
            if entities:
                avg_confidence = np.mean([entity.confidence for entity in entities])
                entity_features['avg_entity_confidence'].append(avg_confidence)
            else:
                entity_features['avg_entity_confidence'].append(0.0)
        
        return {k: np.array(v) for k, v in entity_features.items()}
    
    def get_topic_interpretations(self) -> Dict[int, Dict[str, Any]]:
        """Get interpretable topic information.
        
        Returns:
            Dict mapping topic index to interpretation info
        """
        if not self.topic_analyzer.is_fitted:
            logger.error("Topic model not fitted")
            return {}
        
        topic_words = self.topic_analyzer.get_top_words_per_topic()
        
        # Add topic interpretations based on top words
        topic_interpretations = {}
        for topic_idx, words in topic_words.items():
            # Simple heuristic for topic naming
            topic_name = f"Topic_{topic_idx}"
            
            # Check for pharmacogenomics-related topics
            if any(word in ['drug', 'gene', 'cyp', 'metabolism'] for word in words):
                topic_name = "Pharmacogenomics"
            elif any(word in ['clinical', 'trial', 'patient'] for word in words):
                topic_name = "Clinical Research"
            elif any(word in ['machine', 'learning', 'model', 'algorithm'] for word in words):
                topic_name = "Machine Learning"
            
            topic_interpretations[topic_idx] = {
                'name': topic_name,
                'top_words': words,
                'description': f"Topic focused on {', '.join(words[:3])}"
            }
        
        return topic_interpretations