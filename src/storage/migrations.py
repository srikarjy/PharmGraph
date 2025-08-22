"""Database migration management system."""

import os
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from sqlalchemy import Column, Integer, String, DateTime, Text, MetaData, Table
from sqlalchemy.orm import Session

from ..config import get_logger
from .database import DatabaseManager, get_db_session
from .models import Base


logger = get_logger(__name__)


class Migration:
    """Base class for database migrations."""
    
    def __init__(self, version: str, description: str):
        """Initialize migration.
        
        Args:
            version: Migration version (e.g., '001', '002')
            description: Human-readable description of the migration
        """
        self.version = version
        self.description = description
        self.timestamp = datetime.utcnow()
    
    def up(self, session: Session) -> None:
        """Apply the migration.
        
        Args:
            session: Database session
        """
        raise NotImplementedError("Migration must implement up() method")
    
    def down(self, session: Session) -> None:
        """Rollback the migration.
        
        Args:
            session: Database session
        """
        raise NotImplementedError("Migration must implement down() method")
    
    def __repr__(self):
        return f"<Migration(version='{self.version}', description='{self.description}')>"


class InitialMigration(Migration):
    """Initial migration to create all tables."""
    
    def __init__(self):
        super().__init__('001', 'Create initial database schema')
    
    def up(self, session: Session) -> None:
        """Create all tables defined in models."""
        logger.info("Creating initial database schema")
        Base.metadata.create_all(bind=session.bind)
        logger.info("Initial schema created successfully")
    
    def down(self, session: Session) -> None:
        """Drop all tables."""
        logger.warning("Dropping all database tables")
        Base.metadata.drop_all(bind=session.bind)
        logger.info("All tables dropped")


class AddIndexesMigration(Migration):
    """Migration to add performance indexes."""
    
    def __init__(self):
        super().__init__('002', 'Add performance optimization indexes')
    
    def up(self, session: Session) -> None:
        """Add additional indexes for query optimization."""
        logger.info("Adding performance optimization indexes")
        
        # Add composite indexes for common query patterns
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_variant_gene_quality ON genomic_variants(gene_symbol, quality_score DESC)",
            "CREATE INDEX IF NOT EXISTS idx_association_drug_evidence ON drug_gene_associations(drug_name, evidence_level, quality_score DESC)",
            "CREATE INDEX IF NOT EXISTS idx_quality_score_variant ON quality_scores(variant_id, overall_score DESC)",
            "CREATE INDEX IF NOT EXISTS idx_processing_pipeline_date ON processing_logs(pipeline_name, started_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_report_type_date ON analysis_reports(report_type, created_at DESC)",
        ]
        
        for index_sql in indexes:
            try:
                session.execute(index_sql)
                logger.debug(f"Created index: {index_sql}")
            except Exception as e:
                logger.warning(f"Failed to create index: {index_sql} - {str(e)}")
        
        session.commit()
        logger.info("Performance indexes added successfully")
    
    def down(self, session: Session) -> None:
        """Remove the added indexes."""
        logger.info("Removing performance optimization indexes")
        
        indexes_to_drop = [
            "DROP INDEX IF EXISTS idx_variant_gene_quality",
            "DROP INDEX IF EXISTS idx_association_drug_evidence", 
            "DROP INDEX IF EXISTS idx_quality_score_variant",
            "DROP INDEX IF EXISTS idx_processing_pipeline_date",
            "DROP INDEX IF EXISTS idx_report_type_date",
        ]
        
        for drop_sql in indexes_to_drop:
            try:
                session.execute(drop_sql)
                logger.debug(f"Dropped index: {drop_sql}")
            except Exception as e:
                logger.warning(f"Failed to drop index: {drop_sql} - {str(e)}")
        
        session.commit()
        logger.info("Performance indexes removed")


class MigrationManager:
    """Manages database migrations and schema versioning."""
    
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        """Initialize migration manager.
        
        Args:
            db_manager: Database manager instance. If None, uses global instance.
        """
        self.db_manager = db_manager
        self.migrations: List[Migration] = []
        self._register_migrations()
    
    def _register_migrations(self) -> None:
        """Register all available migrations."""
        self.migrations = [
            InitialMigration(),
            AddIndexesMigration(),
        ]
        
        # Sort migrations by version
        self.migrations.sort(key=lambda m: m.version)
    
    def _ensure_migration_table(self, session: Session) -> None:
        """Ensure the migration tracking table exists."""
        # Create migration table if it doesn't exist
        migration_table_sql = """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            id INTEGER PRIMARY KEY,
            version VARCHAR(50) NOT NULL UNIQUE,
            description TEXT,
            applied_at TIMESTAMP NOT NULL,
            rollback_sql TEXT
        )
        """
        
        session.execute(migration_table_sql)
        session.commit()
    
    def get_applied_migrations(self, session: Session) -> List[str]:
        """Get list of applied migration versions.
        
        Args:
            session: Database session
            
        Returns:
            List[str]: List of applied migration versions
        """
        self._ensure_migration_table(session)
        
        result = session.execute("SELECT version FROM schema_migrations ORDER BY version")
        return [row[0] for row in result.fetchall()]
    
    def get_pending_migrations(self, session: Session) -> List[Migration]:
        """Get list of pending migrations.
        
        Args:
            session: Database session
            
        Returns:
            List[Migration]: List of pending migrations
        """
        applied = set(self.get_applied_migrations(session))
        return [m for m in self.migrations if m.version not in applied]
    
    def apply_migration(self, session: Session, migration: Migration) -> None:
        """Apply a single migration.
        
        Args:
            session: Database session
            migration: Migration to apply
        """
        logger.info(f"Applying migration {migration.version}: {migration.description}")
        
        try:
            # Apply the migration
            migration.up(session)
            
            # Record the migration as applied
            insert_sql = """
            INSERT INTO schema_migrations (version, description, applied_at)
            VALUES (?, ?, ?)
            """
            session.execute(insert_sql, (
                migration.version,
                migration.description,
                datetime.utcnow()
            ))
            
            session.commit()
            logger.info(f"Migration {migration.version} applied successfully")
            
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to apply migration {migration.version}: {str(e)}")
            raise
    
    def rollback_migration(self, session: Session, migration: Migration) -> None:
        """Rollback a single migration.
        
        Args:
            session: Database session
            migration: Migration to rollback
        """
        logger.info(f"Rolling back migration {migration.version}: {migration.description}")
        
        try:
            # Rollback the migration
            migration.down(session)
            
            # Remove the migration record
            delete_sql = "DELETE FROM schema_migrations WHERE version = ?"
            session.execute(delete_sql, (migration.version,))
            
            session.commit()
            logger.info(f"Migration {migration.version} rolled back successfully")
            
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to rollback migration {migration.version}: {str(e)}")
            raise
    
    def migrate(self, target_version: Optional[str] = None) -> None:
        """Apply all pending migrations up to target version.
        
        Args:
            target_version: Target migration version. If None, applies all pending.
        """
        if self.db_manager:
            session = self.db_manager.get_session()
        else:
            session = get_db_session()
        
        try:
            pending = self.get_pending_migrations(session)
            
            if not pending:
                logger.info("No pending migrations to apply")
                return
            
            # Filter migrations up to target version if specified
            if target_version:
                pending = [m for m in pending if m.version <= target_version]
            
            logger.info(f"Applying {len(pending)} pending migrations")
            
            for migration in pending:
                self.apply_migration(session, migration)
            
            logger.info("All migrations applied successfully")
            
        finally:
            session.close()
    
    def rollback(self, target_version: str) -> None:
        """Rollback migrations to target version.
        
        Args:
            target_version: Target version to rollback to
        """
        if self.db_manager:
            session = self.db_manager.get_session()
        else:
            session = get_db_session()
        
        try:
            applied = self.get_applied_migrations(session)
            
            # Find migrations to rollback (in reverse order)
            to_rollback = []
            for version in reversed(applied):
                if version > target_version:
                    migration = next((m for m in self.migrations if m.version == version), None)
                    if migration:
                        to_rollback.append(migration)
            
            if not to_rollback:
                logger.info(f"No migrations to rollback to version {target_version}")
                return
            
            logger.info(f"Rolling back {len(to_rollback)} migrations to version {target_version}")
            
            for migration in to_rollback:
                self.rollback_migration(session, migration)
            
            logger.info(f"Rollback to version {target_version} completed successfully")
            
        finally:
            session.close()
    
    def get_migration_status(self) -> Dict[str, Any]:
        """Get current migration status.
        
        Returns:
            Dict[str, Any]: Migration status information
        """
        if self.db_manager:
            session = self.db_manager.get_session()
        else:
            session = get_db_session()
        
        try:
            applied = self.get_applied_migrations(session)
            pending = self.get_pending_migrations(session)
            
            return {
                'total_migrations': len(self.migrations),
                'applied_count': len(applied),
                'pending_count': len(pending),
                'applied_versions': applied,
                'pending_versions': [m.version for m in pending],
                'current_version': applied[-1] if applied else None,
                'latest_version': self.migrations[-1].version if self.migrations else None
            }
            
        finally:
            session.close()


# Global migration manager instance
_migration_manager = MigrationManager()


def get_migration_manager() -> MigrationManager:
    """Get the global migration manager instance.
    
    Returns:
        MigrationManager: Global migration manager
    """
    return _migration_manager


def migrate_database(target_version: Optional[str] = None) -> None:
    """Apply database migrations using the global manager.
    
    Args:
        target_version: Target migration version. If None, applies all pending.
    """
    _migration_manager.migrate(target_version)


def get_migration_status() -> Dict[str, Any]:
    """Get migration status using the global manager.
    
    Returns:
        Dict[str, Any]: Migration status information
    """
    return _migration_manager.get_migration_status()