"""Feature store integration for online and offline feature serving."""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import sqlite3
import redis
from pathlib import Path
import hashlib
import pickle
from abc import ABC, abstractmethod

from ..config import get_logger, get_config
from .feature_extraction import FeatureSet, FeatureMetadata


logger = get_logger(__name__)


@dataclass
class FeatureDefinition:
    """Definition of a feature in the feature store."""
    name: str
    feature_type: str  # 'numerical', 'categorical', 'embedding'
    description: str
    source: str  # Source system/pipeline
    owner: str
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    version: str = "v1"
    schema: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeatureGroup:
    """Group of related features."""
    name: str
    features: List[str]
    description: str
    source: str
    refresh_interval: timedelta
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class FeatureDriftMetrics:
    """Feature drift detection metrics."""
    feature_name: str
    drift_score: float
    drift_threshold: float
    is_drifting: bool
    detection_method: str
    timestamp: datetime
    reference_stats: Dict[str, float]
    current_stats: Dict[str, float]


class FeatureStoreBackend(ABC):
    """Abstract base class for feature store backends."""
    
    @abstractmethod
    def store_features(self, feature_group: str, features: Dict[str, np.ndarray], 
                      entity_ids: List[str], timestamp: datetime) -> bool:
        """Store features in the backend."""
        pass
    
    @abstractmethod
    def retrieve_features(self, feature_group: str, feature_names: List[str], 
                         entity_ids: List[str]) -> Dict[str, np.ndarray]:
        """Retrieve features from the backend."""
        pass
    
    @abstractmethod
    def list_feature_groups(self) -> List[str]:
        """List available feature groups."""
        pass


class SQLiteFeatureStore(FeatureStoreBackend):
    """SQLite-based feature store for offline features."""
    
    def __init__(self, db_path: str = "feature_store.db"):
        """Initialize SQLite feature store.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Feature definitions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS feature_definitions (
                    name TEXT PRIMARY KEY,
                    feature_type TEXT NOT NULL,
                    description TEXT,
                    source TEXT,
                    owner TEXT,
                    tags TEXT,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP,
                    version TEXT,
                    schema TEXT
                )
            """)
            
            # Feature groups table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS feature_groups (
                    name TEXT PRIMARY KEY,
                    features TEXT,
                    description TEXT,
                    source TEXT,
                    refresh_interval INTEGER,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP
                )
            """)
            
            # Feature values table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS feature_values (
                    feature_group TEXT,
                    entity_id TEXT,
                    feature_name TEXT,
                    feature_value BLOB,
                    timestamp TIMESTAMP,
                    PRIMARY KEY (feature_group, entity_id, feature_name, timestamp)
                )
            """)
            
            # Feature drift metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS feature_drift_metrics (
                    feature_name TEXT,
                    drift_score REAL,
                    drift_threshold REAL,
                    is_drifting BOOLEAN,
                    detection_method TEXT,
                    timestamp TIMESTAMP,
                    reference_stats TEXT,
                    current_stats TEXT,
                    PRIMARY KEY (feature_name, timestamp)
                )
            """)
            
            conn.commit()
    
    def store_features(self, feature_group: str, features: Dict[str, np.ndarray], 
                      entity_ids: List[str], timestamp: datetime) -> bool:
        """Store features in SQLite database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                for feature_name, feature_values in features.items():
                    for entity_id, value in zip(entity_ids, feature_values):
                        # Serialize numpy value
                        serialized_value = pickle.dumps(value)
                        
                        cursor.execute("""
                            INSERT OR REPLACE INTO feature_values 
                            (feature_group, entity_id, feature_name, feature_value, timestamp)
                            VALUES (?, ?, ?, ?, ?)
                        """, (feature_group, entity_id, feature_name, serialized_value, timestamp))
                
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"Failed to store features: {e}")
            return False
    
    def retrieve_features(self, feature_group: str, feature_names: List[str], 
                         entity_ids: List[str]) -> Dict[str, np.ndarray]:
        """Retrieve features from SQLite database."""
        features = {name: [] for name in feature_names}
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                for entity_id in entity_ids:
                    for feature_name in feature_names:
                        cursor.execute("""
                            SELECT feature_value FROM feature_values
                            WHERE feature_group = ? AND entity_id = ? AND feature_name = ?
                            ORDER BY timestamp DESC LIMIT 1
                        """, (feature_group, entity_id, feature_name))
                        
                        result = cursor.fetchone()
                        if result:
                            value = pickle.loads(result[0])
                            features[feature_name].append(value)
                        else:
                            features[feature_name].append(0.0)  # Default value
                
                # Convert to numpy arrays
                return {name: np.array(values) for name, values in features.items()}
                
        except Exception as e:
            logger.error(f"Failed to retrieve features: {e}")
            return {name: np.zeros(len(entity_ids)) for name in feature_names}
    
    def list_feature_groups(self) -> List[str]:
        """List available feature groups."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT DISTINCT feature_group FROM feature_values")
                return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Failed to list feature groups: {e}")
            return []


class RedisFeatureStore(FeatureStoreBackend):
    """Redis-based feature store for online features."""
    
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0,
                 ttl: int = 86400):  # 24 hours default TTL
        """Initialize Redis feature store.
        
        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            ttl: Time-to-live for features in seconds
        """
        self.host = host
        self.port = port
        self.db = db
        self.ttl = ttl
        
        try:
            self.redis_client = redis.Redis(host=host, port=port, db=db, decode_responses=False)
            self.redis_client.ping()
            logger.info(f"Connected to Redis at {host}:{port}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.redis_client = None
    
    def store_features(self, feature_group: str, features: Dict[str, np.ndarray], 
                      entity_ids: List[str], timestamp: datetime) -> bool:
        """Store features in Redis."""
        if not self.redis_client:
            return False
        
        try:
            pipe = self.redis_client.pipeline()
            
            for feature_name, feature_values in features.items():
                for entity_id, value in zip(entity_ids, feature_values):
                    key = f"{feature_group}:{entity_id}:{feature_name}"
                    
                    # Store feature value with metadata
                    feature_data = {
                        'value': pickle.dumps(value),
                        'timestamp': timestamp.isoformat(),
                        'feature_group': feature_group,
                        'feature_name': feature_name
                    }
                    
                    pipe.hset(key, mapping=feature_data)
                    pipe.expire(key, self.ttl)
            
            pipe.execute()
            return True
            
        except Exception as e:
            logger.error(f"Failed to store features in Redis: {e}")
            return False
    
    def retrieve_features(self, feature_group: str, feature_names: List[str], 
                         entity_ids: List[str]) -> Dict[str, np.ndarray]:
        """Retrieve features from Redis."""
        if not self.redis_client:
            return {name: np.zeros(len(entity_ids)) for name in feature_names}
        
        features = {name: [] for name in feature_names}
        
        try:
            pipe = self.redis_client.pipeline()
            
            # Batch retrieve all features
            keys = []
            for entity_id in entity_ids:
                for feature_name in feature_names:
                    key = f"{feature_group}:{entity_id}:{feature_name}"
                    keys.append((key, entity_id, feature_name))
                    pipe.hget(key, 'value')
            
            results = pipe.execute()
            
            # Process results
            for i, (key, entity_id, feature_name) in enumerate(keys):
                result = results[i]
                if result:
                    value = pickle.loads(result)
                    features[feature_name].append(value)
                else:
                    features[feature_name].append(0.0)  # Default value
            
            # Reshape results
            reshaped_features = {}
            for feature_name in feature_names:
                # Extract values for this feature across all entities
                feature_values = []
                for i, entity_id in enumerate(entity_ids):
                    idx = i * len(feature_names) + feature_names.index(feature_name)
                    if idx < len(features[feature_name]):
                        feature_values.append(features[feature_name][idx])
                    else:
                        feature_values.append(0.0)
                
                reshaped_features[feature_name] = np.array(feature_values)
            
            return reshaped_features
            
        except Exception as e:
            logger.error(f"Failed to retrieve features from Redis: {e}")
            return {name: np.zeros(len(entity_ids)) for name in feature_names}
    
    def list_feature_groups(self) -> List[str]:
        """List available feature groups."""
        if not self.redis_client:
            return []
        
        try:
            keys = self.redis_client.keys("*:*:*")
            feature_groups = set()
            
            for key in keys:
                key_str = key.decode('utf-8') if isinstance(key, bytes) else key
                parts = key_str.split(':')
                if len(parts) >= 3:
                    feature_groups.add(parts[0])
            
            return list(feature_groups)
            
        except Exception as e:
            logger.error(f"Failed to list feature groups from Redis: {e}")
            return []


class FeatureDriftDetector:
    """Detects feature drift using statistical methods."""
    
    def __init__(self, drift_threshold: float = 0.1):
        """Initialize drift detector.
        
        Args:
            drift_threshold: Threshold for drift detection
        """
        self.drift_threshold = drift_threshold
    
    def detect_drift(self, reference_data: np.ndarray, current_data: np.ndarray,
                    feature_name: str, method: str = 'ks_test') -> FeatureDriftMetrics:
        """Detect drift between reference and current data.
        
        Args:
            reference_data: Reference/baseline data
            current_data: Current data to compare
            feature_name: Name of the feature
            method: Drift detection method
            
        Returns:
            FeatureDriftMetrics object
        """
        if method == 'ks_test':
            drift_score = self._ks_test_drift(reference_data, current_data)
        elif method == 'psi':
            drift_score = self._psi_drift(reference_data, current_data)
        else:
            drift_score = self._statistical_drift(reference_data, current_data)
        
        is_drifting = drift_score > self.drift_threshold
        
        # Calculate statistics
        reference_stats = {
            'mean': float(np.mean(reference_data)),
            'std': float(np.std(reference_data)),
            'min': float(np.min(reference_data)),
            'max': float(np.max(reference_data))
        }
        
        current_stats = {
            'mean': float(np.mean(current_data)),
            'std': float(np.std(current_data)),
            'min': float(np.min(current_data)),
            'max': float(np.max(current_data))
        }
        
        return FeatureDriftMetrics(
            feature_name=feature_name,
            drift_score=drift_score,
            drift_threshold=self.drift_threshold,
            is_drifting=is_drifting,
            detection_method=method,
            timestamp=datetime.now(),
            reference_stats=reference_stats,
            current_stats=current_stats
        )
    
    def _ks_test_drift(self, reference_data: np.ndarray, current_data: np.ndarray) -> float:
        """Kolmogorov-Smirnov test for drift detection."""
        try:
            from scipy.stats import ks_2samp
            statistic, p_value = ks_2samp(reference_data, current_data)
            return float(statistic)
        except ImportError:
            logger.warning("scipy not available, using statistical drift")
            return self._statistical_drift(reference_data, current_data)
    
    def _psi_drift(self, reference_data: np.ndarray, current_data: np.ndarray) -> float:
        """Population Stability Index for drift detection."""
        # Create bins based on reference data
        bins = np.histogram_bin_edges(reference_data, bins=10)
        
        # Calculate histograms
        ref_hist, _ = np.histogram(reference_data, bins=bins)
        cur_hist, _ = np.histogram(current_data, bins=bins)
        
        # Normalize to probabilities
        ref_prob = ref_hist / np.sum(ref_hist)
        cur_prob = cur_hist / np.sum(cur_hist)
        
        # Avoid division by zero
        ref_prob = np.where(ref_prob == 0, 1e-10, ref_prob)
        cur_prob = np.where(cur_prob == 0, 1e-10, cur_prob)
        
        # Calculate PSI
        psi = np.sum((cur_prob - ref_prob) * np.log(cur_prob / ref_prob))
        return float(psi)
    
    def _statistical_drift(self, reference_data: np.ndarray, current_data: np.ndarray) -> float:
        """Simple statistical drift based on mean and std differences."""
        ref_mean, ref_std = np.mean(reference_data), np.std(reference_data)
        cur_mean, cur_std = np.mean(current_data), np.std(current_data)
        
        # Normalize differences
        mean_diff = abs(cur_mean - ref_mean) / (ref_std + 1e-10)
        std_diff = abs(cur_std - ref_std) / (ref_std + 1e-10)
        
        return float(mean_diff + std_diff)


class FeatureStore:
    """Main feature store interface supporting online and offline serving."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize feature store.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Initialize backends
        self.offline_store = SQLiteFeatureStore(
            db_path=self.config.get('sqlite_path', 'feature_store.db')
        )
        
        # Initialize Redis if configured
        redis_config = self.config.get('redis', {})
        if redis_config.get('enabled', False):
            self.online_store = RedisFeatureStore(
                host=redis_config.get('host', 'localhost'),
                port=redis_config.get('port', 6379),
                db=redis_config.get('db', 0),
                ttl=redis_config.get('ttl', 86400)
            )
        else:
            self.online_store = None
        
        # Initialize drift detector
        self.drift_detector = FeatureDriftDetector(
            drift_threshold=self.config.get('drift_threshold', 0.1)
        )
        
        # Feature definitions registry
        self.feature_definitions = {}
        self.feature_groups = {}
        
        logger.info("FeatureStore initialized")
    
    def register_feature_group(self, feature_group: FeatureGroup) -> bool:
        """Register a feature group.
        
        Args:
            feature_group: Feature group to register
            
        Returns:
            True if successful
        """
        try:
            self.feature_groups[feature_group.name] = feature_group
            
            # Store in SQLite
            with sqlite3.connect(self.offline_store.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO feature_groups 
                    (name, features, description, source, refresh_interval, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    feature_group.name,
                    json.dumps(feature_group.features),
                    feature_group.description,
                    feature_group.source,
                    int(feature_group.refresh_interval.total_seconds()),
                    feature_group.created_at,
                    feature_group.updated_at
                ))
                conn.commit()
            
            logger.info(f"Registered feature group: {feature_group.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register feature group: {e}")
            return False
    
    def register_feature_definition(self, feature_def: FeatureDefinition) -> bool:
        """Register a feature definition.
        
        Args:
            feature_def: Feature definition to register
            
        Returns:
            True if successful
        """
        try:
            self.feature_definitions[feature_def.name] = feature_def
            
            # Store in SQLite
            with sqlite3.connect(self.offline_store.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO feature_definitions 
                    (name, feature_type, description, source, owner, tags, 
                     created_at, updated_at, version, schema)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    feature_def.name,
                    feature_def.feature_type,
                    feature_def.description,
                    feature_def.source,
                    feature_def.owner,
                    json.dumps(feature_def.tags),
                    feature_def.created_at,
                    feature_def.updated_at,
                    feature_def.version,
                    json.dumps(feature_def.schema)
                ))
                conn.commit()
            
            logger.info(f"Registered feature definition: {feature_def.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register feature definition: {e}")
            return False
    
    def store_features_offline(self, feature_group: str, feature_set: FeatureSet) -> bool:
        """Store features for offline training.
        
        Args:
            feature_group: Name of the feature group
            feature_set: Feature set to store
            
        Returns:
            True if successful
        """
        return self.offline_store.store_features(
            feature_group=feature_group,
            features=feature_set.features,
            entity_ids=feature_set.sample_ids,
            timestamp=feature_set.extraction_timestamp
        )
    
    def store_features_online(self, feature_group: str, feature_set: FeatureSet) -> bool:
        """Store features for online serving.
        
        Args:
            feature_group: Name of the feature group
            feature_set: Feature set to store
            
        Returns:
            True if successful
        """
        if not self.online_store:
            logger.warning("Online store not configured")
            return False
        
        return self.online_store.store_features(
            feature_group=feature_group,
            features=feature_set.features,
            entity_ids=feature_set.sample_ids,
            timestamp=feature_set.extraction_timestamp
        )
    
    def get_features_offline(self, feature_group: str, feature_names: List[str], 
                           entity_ids: List[str]) -> Dict[str, np.ndarray]:
        """Retrieve features for offline training.
        
        Args:
            feature_group: Name of the feature group
            feature_names: List of feature names to retrieve
            entity_ids: List of entity IDs
            
        Returns:
            Dict of feature arrays
        """
        return self.offline_store.retrieve_features(feature_group, feature_names, entity_ids)
    
    def get_features_online(self, feature_group: str, feature_names: List[str], 
                          entity_ids: List[str]) -> Dict[str, np.ndarray]:
        """Retrieve features for online serving.
        
        Args:
            feature_group: Name of the feature group
            feature_names: List of feature names to retrieve
            entity_ids: List of entity IDs
            
        Returns:
            Dict of feature arrays
        """
        if not self.online_store:
            logger.warning("Online store not configured, falling back to offline")
            return self.get_features_offline(feature_group, feature_names, entity_ids)
        
        return self.online_store.retrieve_features(feature_group, feature_names, entity_ids)
    
    def monitor_feature_drift(self, feature_group: str, feature_names: List[str],
                            reference_window_days: int = 30) -> List[FeatureDriftMetrics]:
        """Monitor feature drift over time.
        
        Args:
            feature_group: Name of the feature group
            feature_names: List of feature names to monitor
            reference_window_days: Days to look back for reference data
            
        Returns:
            List of drift metrics
        """
        drift_metrics = []
        
        try:
            # Get reference data (older)
            reference_date = datetime.now() - timedelta(days=reference_window_days)
            
            # Get current data (recent)
            current_date = datetime.now() - timedelta(days=1)
            
            with sqlite3.connect(self.offline_store.db_path) as conn:
                cursor = conn.cursor()
                
                for feature_name in feature_names:
                    # Get reference data
                    cursor.execute("""
                        SELECT feature_value FROM feature_values
                        WHERE feature_group = ? AND feature_name = ? 
                        AND timestamp BETWEEN ? AND ?
                    """, (feature_group, feature_name, 
                         reference_date - timedelta(days=7), reference_date))
                    
                    ref_results = cursor.fetchall()
                    if not ref_results:
                        continue
                    
                    reference_data = np.array([pickle.loads(row[0]) for row in ref_results])
                    
                    # Get current data
                    cursor.execute("""
                        SELECT feature_value FROM feature_values
                        WHERE feature_group = ? AND feature_name = ? 
                        AND timestamp >= ?
                    """, (feature_group, feature_name, current_date))
                    
                    cur_results = cursor.fetchall()
                    if not cur_results:
                        continue
                    
                    current_data = np.array([pickle.loads(row[0]) for row in cur_results])
                    
                    # Detect drift
                    drift_metric = self.drift_detector.detect_drift(
                        reference_data, current_data, feature_name
                    )
                    
                    drift_metrics.append(drift_metric)
                    
                    # Store drift metrics
                    cursor.execute("""
                        INSERT INTO feature_drift_metrics 
                        (feature_name, drift_score, drift_threshold, is_drifting, 
                         detection_method, timestamp, reference_stats, current_stats)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        drift_metric.feature_name,
                        drift_metric.drift_score,
                        drift_metric.drift_threshold,
                        drift_metric.is_drifting,
                        drift_metric.detection_method,
                        drift_metric.timestamp,
                        json.dumps(drift_metric.reference_stats),
                        json.dumps(drift_metric.current_stats)
                    ))
                
                conn.commit()
        
        except Exception as e:
            logger.error(f"Failed to monitor feature drift: {e}")
        
        return drift_metrics
    
    def get_feature_statistics(self, feature_group: str, 
                             feature_names: List[str]) -> Dict[str, Dict[str, float]]:
        """Get feature statistics.
        
        Args:
            feature_group: Name of the feature group
            feature_names: List of feature names
            
        Returns:
            Dict of feature statistics
        """
        statistics = {}
        
        try:
            with sqlite3.connect(self.offline_store.db_path) as conn:
                cursor = conn.cursor()
                
                for feature_name in feature_names:
                    cursor.execute("""
                        SELECT feature_value FROM feature_values
                        WHERE feature_group = ? AND feature_name = ?
                        ORDER BY timestamp DESC LIMIT 1000
                    """, (feature_group, feature_name))
                    
                    results = cursor.fetchall()
                    if results:
                        values = np.array([pickle.loads(row[0]) for row in results])
                        
                        statistics[feature_name] = {
                            'count': len(values),
                            'mean': float(np.mean(values)),
                            'std': float(np.std(values)),
                            'min': float(np.min(values)),
                            'max': float(np.max(values)),
                            'median': float(np.median(values)),
                            'p25': float(np.percentile(values, 25)),
                            'p75': float(np.percentile(values, 75))
                        }
        
        except Exception as e:
            logger.error(f"Failed to get feature statistics: {e}")
        
        return statistics